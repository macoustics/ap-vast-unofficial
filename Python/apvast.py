import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import libdetectability as ld

EXPERIMENTAL_NORMALIZE_GAINS = True
EXPERIMENTAL_REGULARIZATION = True

def approx(a, b, rtol=1e-5, atol=1e-15, etol=1e-25):
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape
        for ia, ib in zip(a.flatten(), b.flatten()):
            assert abs(ia - ib) <= atol, f"{ia} and {ib} fail atol"
            assert abs(ia - ib) / (abs(ib) + etol) <= rtol, f"{ia} and {ib} fail rtol"
    else:
        assert abs(a - b) / (abs(b) + etol) <= rtol
        assert abs(a - b) <= atol

# joint diagoanlization
def jdiag(A, B):
    # throws on non-semidefinite B
    if EXPERIMENTAL_REGULARIZATION:
        reg = 1e-7
        Bc = np.linalg.cholesky(B + reg * np.eye(B.shape[0]))
    else:
        lmax = np.linalg.norm(B, ord=2)
        Bc = np.linalg.cholesky(B + 1e-8 * lmax * np.eye(B.shape[0]))
    C0 = sp.linalg.solve_triangular(Bc, A, lower=True)
    C1 = sp.linalg.solve_triangular(np.conj(Bc), C0.T, lower=True).T
    [T, U] = sp.linalg.schur(C1)
    X = sp.linalg.solve_triangular(np.conj(Bc).T, U, lower=False)
    dind = np.flip(np.argsort(np.diag(T)))
    dd = np.diag(T)[dind]
    D = np.diag(dd)
    U = X[:, dind]
    return U, D

# implementation of the MIM version of AP-VAST in python
class apvast:
    def __init__(self, \
            block_size: int, \
            rir_A, \
            rir_B, \
            filter_length: int, \
            modeling_delay: int, \
            reference_index_A: int, \
            reference_index_B: int, \
            number_of_eigenvectors: int, \
            mu: float, \
            statistics_buffer_length: int,
            hop_size: int = None, \
            sampling_rate: int = 48000, \
            run_A: bool = True,
            run_B: bool = True,
            perceptual: bool = True,
        ):

        # map input params
        self.block_size = block_size
        self.rir_A = rir_A 
        self.rir_B = rir_B
        self.filter_length = filter_length
        self.modeling_delay = modeling_delay
        self.reference_index_A = reference_index_A
        self.reference_index_B = reference_index_B
        self.number_of_eigenvectors = number_of_eigenvectors
        self.mu = mu
        self.sampling_rate = sampling_rate
        self.statistics_buffer_length = statistics_buffer_length
        self.run_A = run_A
        self.run_B = run_B

        # create perceptual model
        self.perceptual = perceptual
        if self.perceptual:
            np.seterr(divide='ignore')
            self.model = ld.Detectability(frame_size=self.block_size,
                                          sampling_rate=self.sampling_rate,
                                          taps=32,
                                          dbspl=60.0, # Note: relax_threshold so ignored
                                          spl=1.0e-3, # Note: relax_threshold so ignored
                                          relax_threshold=True,
                                          )

        # validate
        if self.block_size % 2 != 0:
            raise RuntimeError("block size must be modulo 2")

        if rir_A.shape != rir_B.shape:
            raise RuntimeError("rirs of unequal size")

        # calculate remaining params
        self.hop_size = hop_size if hop_size else self.block_size // 2
        self.window = np.sin(np.pi / self.block_size * np.arange(self.block_size)).reshape(-1, 1)
        self.input_A_block = np.zeros((self.block_size, 1))
        self.input_B_block = np.zeros((self.block_size, 1))
        self.rir_length = rir_A.shape[0]
        self.number_of_srcs = rir_A.shape[1]
        self.number_of_mics = rir_A.shape[2]

        # calculate target rirs
        self.target_rir_A = np.zeros((self.rir_length, self.number_of_mics))
        self.target_rir_B = np.zeros((self.rir_length, self.number_of_mics))
        for m in range(self.number_of_mics):
            self.target_rir_A[:,m] = np.concatenate([
                np.zeros((self.modeling_delay)), 
                rir_A[:self.rir_length - self.modeling_delay, self.reference_index_A, m]
            ])
            self.target_rir_B[:,m] = np.concatenate([
                np.zeros((self.modeling_delay)), 
                rir_B[:self.rir_length - self.modeling_delay, self.reference_index_B, m]
            ])

        # pre-alloc states
        self.rir_A_to_A_state = np.zeros((self.rir_length - 1, self.number_of_srcs, self.number_of_mics))
        self.rir_A_to_B_state = np.zeros((self.rir_length - 1, self.number_of_srcs, self.number_of_mics))
        self.target_rir_A_to_A_state = np.zeros((self.rir_length - 1, self.number_of_mics))
        self.rir_B_to_A_state = np.zeros((self.rir_length - 1, self.number_of_srcs, self.number_of_mics))
        self.rir_B_to_B_state = np.zeros((self.rir_length - 1, self.number_of_srcs, self.number_of_mics))
        self.target_rir_B_to_B_state = np.zeros((self.rir_length - 1, self.number_of_mics))

        # init loudspeaker response buffers
        # init with noise for numerical reasons
        self.loudspeaker_response_A_to_A_buffer = 1e-3 * np.random.randn(self.block_size, self.number_of_srcs, self.number_of_mics)
        self.loudspeaker_response_A_to_B_buffer = 1e-3 * np.random.randn(self.block_size, self.number_of_srcs, self.number_of_mics)
        self.loudspeaker_response_B_to_A_buffer = 1e-3 * np.random.randn(self.block_size, self.number_of_srcs, self.number_of_mics)
        self.loudspeaker_response_B_to_B_buffer = 1e-3 * np.random.randn(self.block_size, self.number_of_srcs, self.number_of_mics)
        self.loudspeaker_target_response_A_to_A_buffer = 1e-3 * np.random.randn(self.block_size, self.number_of_mics)
        self.loudspeaker_target_response_B_to_B_buffer = 1e-3 * np.random.randn(self.block_size, self.number_of_mics)

        # init loudspeaker response overlap buffers
        self.loudspeaker_weighted_response_A_to_A_overlap_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_response_A_to_B_overlap_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_response_B_to_A_overlap_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_response_B_to_B_overlap_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer = np.zeros((self.block_size, self.number_of_mics))
        self.loudspeaker_weighted_target_response_B_to_B_overlap_buffer = np.zeros((self.block_size, self.number_of_mics))

        # init loudspeaker response overlap buffers
        self.loudspeaker_weighted_response_A_to_A_buffer = np.zeros((self.statistics_buffer_length, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_response_A_to_B_buffer = np.zeros((self.statistics_buffer_length, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_response_B_to_A_buffer = np.zeros((self.statistics_buffer_length, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_response_B_to_B_buffer = np.zeros((self.statistics_buffer_length, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_weighted_target_response_A_to_A_buffer = np.zeros((self.statistics_buffer_length, self.number_of_mics))
        self.loudspeaker_weighted_target_response_B_to_B_buffer = np.zeros((self.statistics_buffer_length, self.number_of_mics))

        # init output overlap buffers
        self.output_A_overlap_buffer = np.zeros((self.number_of_eigenvectors, self.block_size, self.number_of_srcs))
        self.output_B_overlap_buffer = np.zeros((self.number_of_eigenvectors, self.block_size, self.number_of_srcs))
        self.output_A_t_overlap_buffer = np.zeros((self.number_of_eigenvectors, self.block_size, self.number_of_srcs))
        self.output_B_t_overlap_buffer = np.zeros((self.number_of_eigenvectors, self.block_size, self.number_of_srcs))

    def process_input_buffers(self, input_A, input_B):
        if input_A.size != self.hop_size or input_A.size != self.hop_size:
            raise RuntimeError("invalid input size")

        self.update_loudspeaker_response_buffers(input_A, input_B)
        self.update_weighted_target_signals()
        self.update_weighted_loudspeaker_response()
        self.update_statistics()
        self.calculate_filter_spectra(self.mu)
        self.update_input_blocks(input_A, input_B)
        output_buffer_A, output_buffer_B, output_buffer_A_t, output_buffer_B_t = self.compute_output_buffers()

        return output_buffer_A, output_buffer_B, output_buffer_A_t, output_buffer_B_t

    def update_loudspeaker_response_buffers(self, input_A, input_B):
        idx = np.array([i for i in range(self.hop_size, self.block_size)]) # NOTE: could be wrong?
        for m in range(self.number_of_mics):
            # always run as target required by both
            tmp_input, tmp_state = sp.signal.lfilter(self.target_rir_A[:, m], 1, input_A, zi=self.target_rir_A_to_A_state[:, m])
            self.target_rir_A_to_A_state[:, m] = tmp_state
            self.loudspeaker_target_response_A_to_A_buffer[:, m] = np.concatenate([self.loudspeaker_target_response_A_to_A_buffer[idx, m], tmp_input])

            tmp_input, tmp_state = sp.signal.lfilter(self.target_rir_B[:, m], 1, input_B, zi=self.target_rir_B_to_B_state[:, m])
            self.target_rir_B_to_B_state[:, m] = tmp_state
            self.loudspeaker_target_response_B_to_B_buffer[:, m] = np.concatenate([self.loudspeaker_target_response_B_to_B_buffer[idx, m], tmp_input])

            for l in range(self.number_of_srcs):
                tmp_input, tmp_state = sp.signal.lfilter(self.rir_A[:, l, m], 1, input_A, zi=self.rir_A_to_A_state[:, l, m])
                self.rir_A_to_A_state[:, l, m] = tmp_state
                self.loudspeaker_response_A_to_A_buffer[:, l, m] = np.concatenate([self.loudspeaker_response_A_to_A_buffer[idx, l, m], tmp_input])

                tmp_input, tmp_state = sp.signal.lfilter(self.rir_B[:, l, m], 1, input_A, zi=self.rir_A_to_B_state[:, l, m])
                self.rir_A_to_B_state[:, l, m] = tmp_state
                self.loudspeaker_response_A_to_B_buffer[:, l, m] = np.concatenate([self.loudspeaker_response_A_to_B_buffer[idx, l, m], tmp_input])

                tmp_input, tmp_state = sp.signal.lfilter(self.rir_A[:, l, m], 1, input_B, zi=self.rir_B_to_A_state[:, l, m])
                self.rir_B_to_A_state[:, l, m] = tmp_state
                self.loudspeaker_response_B_to_A_buffer[:, l, m] = np.concatenate([self.loudspeaker_response_B_to_A_buffer[idx, l, m], tmp_input])

                tmp_input, tmp_state = sp.signal.lfilter(self.rir_B[:, l, m], 1, input_B, zi=self.rir_B_to_B_state[:, l, m])
                self.rir_B_to_B_state[:, l, m] = tmp_state
                self.loudspeaker_response_B_to_B_buffer[:, l, m] = np.concatenate([self.loudspeaker_response_B_to_B_buffer[idx, l, m], tmp_input])


    def update_weighted_target_signals(self):
        # calculate spectra
        target_A_to_A_spectra = np.zeros((self.block_size // 2 + 1, self.number_of_mics), dtype=complex)
        target_B_to_B_spectra = np.zeros((self.block_size // 2 + 1, self.number_of_mics), dtype=complex)
        for m in range(self.number_of_mics):
            target_A_to_A_spectra[:, m] = np.fft.rfft(np.multiply(self.window.squeeze(1), self.loudspeaker_target_response_A_to_A_buffer[:, m]), axis=0)
            target_B_to_B_spectra[:, m] = np.fft.rfft(np.multiply(self.window.squeeze(1), self.loudspeaker_target_response_B_to_B_buffer[:, m]), axis=0)

        self.update_perceptual_weighting(target_A_to_A_spectra, target_B_to_B_spectra)

        # circular convolution with weighting filter
        target_A_to_A_spectra = np.multiply(target_A_to_A_spectra, self.weighting_spectra_A)
        target_B_to_B_spectra = np.multiply(target_B_to_B_spectra, self.weighting_spectra_B)

        # WOLA reconstruction
        for m in range(self.number_of_mics):
            # zone A
            tmp_old = self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[:, m]
            tmp_new = np.multiply(self.window.squeeze(1), np.fft.irfft(target_A_to_A_spectra[:, m], axis=0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[:, m] = np.pad(tmp_old[self.hop_size: self.block_size], (0, self.hop_size)) + tmp_new

            # Zone B
            tmp_old = self.loudspeaker_weighted_target_response_B_to_B_overlap_buffer[:, m]
            tmp_new = np.multiply(self.window.squeeze(1), np.fft.irfft(target_B_to_B_spectra[:, m], axis=0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_target_response_B_to_B_overlap_buffer[:, m] = np.pad(tmp_old[self.hop_size: self.block_size], (0, self.hop_size)) + tmp_new

        # update weighted_target_response_buffers
        idx = np.array([i for i in range(self.hop_size, self.statistics_buffer_length)]) # NOTE: could be wrong?
        for m in range(self.number_of_mics):
            self.loudspeaker_weighted_target_response_A_to_A_buffer[:, m] = np.concatenate([
                self.loudspeaker_weighted_target_response_A_to_A_buffer[idx, m], 
                self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[0:self.hop_size, m]])
            self.loudspeaker_weighted_target_response_B_to_B_buffer[:, m] = np.concatenate([
                self.loudspeaker_weighted_target_response_B_to_B_buffer[idx, m], 
                self.loudspeaker_weighted_target_response_B_to_B_overlap_buffer[0:self.hop_size, m]])

    def update_weighted_loudspeaker_response(self):
        # calculate spectra
        A_to_A_spectra = np.zeros((self.block_size // 2 + 1, self.number_of_srcs, self.number_of_mics), dtype=complex)
        A_to_B_spectra = np.zeros((self.block_size // 2 + 1, self.number_of_srcs, self.number_of_mics), dtype=complex)
        B_to_A_spectra = np.zeros((self.block_size // 2 + 1, self.number_of_srcs, self.number_of_mics), dtype=complex)
        B_to_B_spectra = np.zeros((self.block_size // 2 + 1, self.number_of_srcs, self.number_of_mics), dtype=complex)

        for m in range(self.number_of_mics):
            if self.run_A:
                tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_A_to_A_buffer[:, :, m]
                A_to_A_spectra[:, :, m] = np.fft.rfft(tmp, n=self.block_size, axis=0)
                tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_A_to_B_buffer[:, :, m]
                A_to_B_spectra[:, :, m] = np.fft.rfft(tmp, n=self.block_size, axis=0)

            if self.run_B:
                tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_B_to_B_buffer[:, :, m]
                B_to_B_spectra[:, :, m] = np.fft.rfft(tmp, n=self.block_size, axis=0)
                tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_B_to_A_buffer[:, :, m]
                B_to_A_spectra[:, :, m] = np.fft.rfft(tmp, n=self.block_size, axis=0)

        # circular convolution with weighting filter
        for m in range(self.number_of_mics):
            A_to_A_spectra[:, :, m] = np.multiply(A_to_A_spectra[:, :, m], np.tile(self.weighting_spectra_A[:, m].reshape(-1, 1), (1, self.number_of_srcs)))
            A_to_B_spectra[:, :, m] = np.multiply(A_to_B_spectra[:, :, m], np.tile(self.weighting_spectra_B[:, m].reshape(-1, 1), (1, self.number_of_srcs)))
            B_to_A_spectra[:, :, m] = np.multiply(B_to_A_spectra[:, :, m], np.tile(self.weighting_spectra_A[:, m].reshape(-1, 1), (1, self.number_of_srcs)))
            B_to_B_spectra[:, :, m] = np.multiply(B_to_B_spectra[:, :, m], np.tile(self.weighting_spectra_B[:, m].reshape(-1, 1), (1, self.number_of_srcs)))

        # WOLA reconstruction
        idx = np.array([i for i in range(self.hop_size, self.block_size)]) # NOTE: could be wrong?
        for m in range(self.number_of_mics):
            # signal A to zone A
            tmp_old = self.loudspeaker_weighted_response_A_to_A_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.irfft(A_to_A_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_A_to_A_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal A to zone B
            tmp_old = self.loudspeaker_weighted_response_A_to_B_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.irfft(A_to_B_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_A_to_B_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal B to zone A
            tmp_old = self.loudspeaker_weighted_response_B_to_A_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.irfft(B_to_A_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_B_to_A_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal B to zone B
            tmp_old = self.loudspeaker_weighted_response_B_to_B_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.irfft(B_to_B_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_B_to_B_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

        # update weighted_target_response_buffers
        idx = np.array([i for i in range(self.hop_size, self.statistics_buffer_length)]) # NOTE: could be wrong?
        for m in range(self.number_of_mics):
            self.loudspeaker_weighted_response_A_to_A_buffer[:(self.statistics_buffer_length - self.hop_size), :, m] = self.loudspeaker_weighted_response_A_to_A_buffer[idx, :, m]
            self.loudspeaker_weighted_response_A_to_A_buffer[(self.statistics_buffer_length - self.hop_size):, :, m] = self.loudspeaker_weighted_response_A_to_A_overlap_buffer[0:self.hop_size, :, m]

            self.loudspeaker_weighted_response_A_to_B_buffer[:, :, m] = np.concatenate([
                self.loudspeaker_weighted_response_A_to_B_buffer[idx, :, m], 
                self.loudspeaker_weighted_response_A_to_B_overlap_buffer[0:self.hop_size, :, m]])

            self.loudspeaker_weighted_response_B_to_A_buffer[:, :, m] = np.concatenate([
                self.loudspeaker_weighted_response_B_to_A_buffer[idx, :, m], 
                self.loudspeaker_weighted_response_B_to_A_overlap_buffer[0:self.hop_size, :, m]])

            self.loudspeaker_weighted_response_B_to_B_buffer[:, :, m] = np.concatenate([
                self.loudspeaker_weighted_response_B_to_B_buffer[idx, :, m], 
                self.loudspeaker_weighted_response_B_to_B_overlap_buffer[0:self.hop_size, :, m]])

    def update_perceptual_weighting(self, target_A_to_A_spectra,  target_B_to_B_spectra):
        if self.perceptual:
            self.weighting_spectra_A = np.zeros((self.block_size // 2 + 1, self.number_of_mics), dtype=complex)
            self.weighting_spectra_B = np.zeros((self.block_size // 2 + 1, self.number_of_mics), dtype=complex)
            for i in range(self.number_of_mics):
                self.weighting_spectra_A[:,i] = self.model.gain(np.fft.irfft(target_A_to_A_spectra[:,i]))
                self.weighting_spectra_B[:,i] = self.model.gain(np.fft.irfft(target_B_to_B_spectra[:,i]))

                # TODO: Check if this does something nice... we normalize all gains
                if EXPERIMENTAL_NORMALIZE_GAINS:
                    self.weighting_spectra_A[:,i] = self.weighting_spectra_A[:,i] / np.linalg.norm(self.weighting_spectra_A[:,i])
                    self.weighting_spectra_B[:,i] = self.weighting_spectra_B[:,i] / np.linalg.norm(self.weighting_spectra_B[:,i])
        else:
            self.weighting_spectra_A = np.ones((self.block_size // 2 + 1, self.number_of_mics)) + 1.0j * np.zeros((self.block_size // 2 + 1, self.number_of_mics))
            self.weighting_spectra_B = np.ones((self.block_size // 2 + 1, self.number_of_mics)) + 1.0j * np.zeros((self.block_size // 2 + 1, self.number_of_mics))

    def update_statistics(self):
        self.reset_statistics()

        for m in range(self.number_of_mics):
            if self.run_A:
                Y = np.zeros((self.filter_length * self.number_of_srcs, self.statistics_buffer_length - self.filter_length))
                for s in range(self.number_of_srcs):
                    Y[s * self.filter_length: (s + 1) * self.filter_length, :] = sp.linalg.toeplitz(
                            np.flipud(self.loudspeaker_weighted_response_A_to_A_buffer[0:self.filter_length, s, m]),
                            self.loudspeaker_weighted_response_A_to_A_buffer[self.filter_length:, s, m])
                self.R_A_to_A += Y @ Y.T
                self.r_A += Y @ self.loudspeaker_weighted_target_response_A_to_A_buffer[self.filter_length:, m].reshape(-1, 1)

                Y = np.zeros((self.filter_length * self.number_of_srcs, self.statistics_buffer_length - self.filter_length))
                for s in range(self.number_of_srcs):
                    Y[s * self.filter_length: (s + 1) * self.filter_length, :] = sp.linalg.toeplitz(
                            np.flipud(self.loudspeaker_weighted_response_A_to_B_buffer[0:self.filter_length, s, m]),
                            self.loudspeaker_weighted_response_A_to_B_buffer[self.filter_length:, s, m])
                self.R_A_to_B += Y @ Y.T

            if self.run_B:
                Y = np.zeros((self.filter_length * self.number_of_srcs, self.statistics_buffer_length - self.filter_length))
                for s in range(self.number_of_srcs):
                    Y[s * self.filter_length: (s + 1) * self.filter_length, :] = sp.linalg.toeplitz(
                            np.flipud(self.loudspeaker_weighted_response_B_to_B_buffer[0:self.filter_length, s, m]),
                            self.loudspeaker_weighted_response_B_to_B_buffer[self.filter_length:, s, m])
                self.R_B_to_B += Y @ Y.T
                self.r_B += Y @ self.loudspeaker_weighted_target_response_B_to_B_buffer[self.filter_length:, m].reshape(-1, 1)


                Y = np.zeros((self.filter_length * self.number_of_srcs, self.statistics_buffer_length - self.filter_length))
                for s in range(self.number_of_srcs):
                    Y[s * self.filter_length: (s + 1) * self.filter_length, :] = sp.linalg.toeplitz(
                            np.flipud(self.loudspeaker_weighted_response_B_to_A_buffer[0:self.filter_length, s, m]),
                            self.loudspeaker_weighted_response_B_to_A_buffer[self.filter_length:, s, m])
                self.R_B_to_A += Y @ Y.T

    def reset_statistics(self):
        if self.run_A:
            self.R_A_to_A = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
            self.R_A_to_B = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        if self.run_B:
            self.R_B_to_A = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
            self.R_B_to_B = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        if self.run_A:
            self.r_A = np.zeros((self.filter_length * self.number_of_srcs, 1))
        if self.run_B:
            self.r_B = np.zeros((self.filter_length * self.number_of_srcs, 1))

    def calculate_filter_spectra(self, mu):
        if self.run_A:
            self.U_A, self.lambda_A = jdiag(self.R_A_to_A, self.R_A_to_B)
        if self.run_B:
            self.U_B, self.lambda_B = jdiag(self.R_B_to_B, self.R_B_to_A)

        if self.run_A:
            self.lambda_A = np.diag(self.lambda_A)
        if self.run_B:
            self.lambda_B = np.diag(self.lambda_B)

        filter_target = np.zeros(self.filter_length * self.number_of_srcs)
        filter_target[self.filter_length * self.reference_index_A + self.modeling_delay] = 1.0

        if self.run_A:
            self.w_A = np.zeros((self.number_of_eigenvectors, self.filter_length * self.number_of_srcs, 1))
            self.filter_spectra_A = []
        else:
            self.w_A = None
        if self.run_B:
            self.w_B = np.zeros((self.number_of_eigenvectors, self.filter_length * self.number_of_srcs, 1))
            self.filter_spectra_B = []
        else:
            self.w_B = None

        self.filter_spectra_A_t = []
        self.filter_spectra_B_t = []

        for i in range(self.number_of_eigenvectors):
            if self.run_A:
                if i > 0:
                    self.w_A[i] = self.w_A[i - 1]
                self.w_A[i] = np.add(self.w_A[i], np.multiply(np.inner(self.U_A[:, i], self.r_A.squeeze(-1)) / (self.lambda_A[i] + mu), self.U_A[:, i].reshape(-1, 1)))
            if self.run_B:
                if i > 0:
                    self.w_B[i] = self.w_B[i - 1]
                self.w_B[i] = np.add(self.w_B[i], np.multiply(np.inner(self.U_B[:, i], self.r_B.squeeze(-1)) / (self.lambda_B[i] + mu), self.U_B[:, i].reshape(-1, 1)))

            if self.run_A:
                self.filter_spectra_A.append(np.fft.rfft(np.reshape(self.w_A[i], (self.filter_length, self.number_of_srcs, 1), order='F'), self.block_size, 0).squeeze(2))
            self.filter_spectra_A_t.append(np.fft.rfft(np.reshape(filter_target, (self.filter_length, self.number_of_srcs, 1), order='F'), self.block_size, 0).squeeze(2))

            if self.run_B:
                self.filter_spectra_B.append(np.fft.rfft(np.reshape(self.w_B[i], (self.filter_length, self.number_of_srcs, 1), order='F'), self.block_size, 0).squeeze(2))
            self.filter_spectra_B_t.append(np.fft.rfft(np.reshape(filter_target, (self.filter_length, self.number_of_srcs, 1), order='F'), self.block_size, 0).squeeze(2))

    def update_input_blocks(self, input_A, input_B):
        self.input_A_block = np.concatenate([self.input_A_block.squeeze(1)[self.hop_size : self.block_size], input_A]).reshape(-1, 1)
        self.input_B_block = np.concatenate([self.input_B_block.squeeze(1)[self.hop_size : self.block_size], input_B]).reshape(-1, 1)

    def compute_output_buffers(self):
        # compute input spectra
        self.input_spectrum_A = np.fft.rfft(np.multiply(self.window.squeeze(1), self.input_A_block.squeeze(1)), axis=0).reshape(-1, 1)
        self.input_spectrum_B = np.fft.rfft(np.multiply(self.window.squeeze(1), self.input_B_block.squeeze(1)), axis=0).reshape(-1, 1)

        if self.run_A:
            output_buffer_A = []
        else:
            output_buffer_A = None
        output_buffer_A_t = []

        if self.run_B:
            output_buffer_B = []
        else:
            output_buffer_B = None
        output_buffer_B_t = []

        for i in range(self.number_of_eigenvectors):
            # circular convolution with the filter spectra
            if self.run_A:
                output_spectra_A = np.multiply(np.tile(self.input_spectrum_A, (1, self.number_of_srcs)), self.filter_spectra_A[i])
            output_spectra_A_t = np.multiply(np.tile(self.input_spectrum_A, (1, self.number_of_srcs)), self.filter_spectra_A_t[i])
            if self.run_B:
                output_spectra_B = np.multiply(np.tile(self.input_spectrum_B, (1, self.number_of_srcs)), self.filter_spectra_B[i])
            output_spectra_B_t = np.multiply(np.tile(self.input_spectrum_B, (1, self.number_of_srcs)), self.filter_spectra_B_t[i])

            # update the output overlap buffers
            idx = np.arange(self.hop_size, self.block_size)
            if self.run_A:
                self.output_A_overlap_buffer[i,:,:] = np.concatenate([
                            self.output_A_overlap_buffer[i, idx, :], 
                            np.zeros((self.hop_size, self.number_of_srcs))
                        ]) 
                tmp = np.fft.irfft(output_spectra_A, self.block_size, axis=0)
                assert np.linalg.norm(np.imag(tmp)) < 1e-8
                tmp = np.real(tmp)
                tmp = np.multiply(tmp, np.tile(self.window, (1, self.number_of_srcs)))
                self.output_A_overlap_buffer[i,:,:] += tmp 

            self.output_A_t_overlap_buffer[i,:,:] = np.concatenate([
                        self.output_A_t_overlap_buffer[i, idx, :], 
                        np.zeros((self.hop_size, self.number_of_srcs))
                    ]) 
            tmp = np.fft.irfft(output_spectra_A_t, self.block_size, axis=0)
            assert np.linalg.norm(np.imag(tmp)) < 1e-8
            tmp = np.real(tmp)
            tmp = np.multiply(tmp, np.tile(self.window, (1, self.number_of_srcs)))
            self.output_A_t_overlap_buffer[i,:,:] += tmp 

            if self.run_B:
                self.output_B_overlap_buffer[i,:,:] = np.concatenate([
                            self.output_B_overlap_buffer[i, idx, :], 
                            np.zeros((self.hop_size, self.number_of_srcs))
                        ]) 
                tmp = np.fft.irfft(output_spectra_B, self.block_size, axis=0)
                assert np.linalg.norm(np.imag(tmp)) < 1e-8
                tmp = np.real(tmp)
                tmp = np.multiply(tmp, np.tile(self.window, (1, self.number_of_srcs)))
                self.output_B_overlap_buffer[i,:,:] += tmp 

            self.output_B_t_overlap_buffer[i,:,:] = np.concatenate([
                        self.output_B_t_overlap_buffer[i, idx, :], 
                        np.zeros((self.hop_size, self.number_of_srcs))
                    ]) 
            tmp = np.fft.irfft(output_spectra_B_t, self.block_size, axis=0)
            assert np.linalg.norm(np.imag(tmp)) < 1e-8
            tmp = np.real(tmp)
            tmp = np.multiply(tmp, np.tile(self.window, (1, self.number_of_srcs)))
            self.output_B_t_overlap_buffer[i,:,:] += tmp 

            # extract samples for the output buffers
            if self.run_A:
                output_buffer_A.append(self.output_A_overlap_buffer[i, :self.hop_size, :])
            output_buffer_A_t.append(self.output_A_t_overlap_buffer[i, :self.hop_size, :])
            if self.run_B:
                output_buffer_B.append(self.output_B_overlap_buffer[i, :self.hop_size, :])
            output_buffer_B_t.append(self.output_B_t_overlap_buffer[i, :self.hop_size, :])

        return output_buffer_A, output_buffer_B, output_buffer_A_t, output_buffer_B_t

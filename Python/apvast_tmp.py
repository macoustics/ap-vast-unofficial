import numpy as np 
import scipy as sp


def approx(a, b, rtol=1e-5, atol=1e-15, etol=1e-25):
    assert a.shape == b.shape
    for ia, ib in zip(np.nditer(a), np.nditer(b)):
        assert abs(ia - ib) / abs(ib + etol) <= rtol
        assert abs(ia - ib) <= atol

# joint diagoanlization
def jdiag(A, B):
    # throws on non-semidefinite B
    Bc = np.linalg.cholesky(B)
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
            hop_size: int = None \
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
        self.statistics_buffer_length = statistics_buffer_length

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
        self.loudspeaker_response_A_to_A_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_response_A_to_B_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_response_B_to_A_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_response_B_to_B_buffer = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics))
        self.loudspeaker_target_response_A_to_A_buffer = np.zeros((self.block_size, self.number_of_mics))
        self.loudspeaker_target_response_B_to_B_buffer = np.zeros((self.block_size, self.number_of_mics))

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
        self.output_A_overlap_buffer = np.zeros((self.block_size, self.number_of_srcs))
        self.output_B_overlap_buffer = np.zeros((self.block_size, self.number_of_srcs))

    def process_input_buffers(self, input_A, input_B):
        if input_A.size != self.hop_size or input_A.size != self.hop_size:
            raise RuntimeError("invalid input size")

        self.update_loudspeaker_response_buffers(input_A, input_B)
        self.update_weighted_target_signals()
        self.update_weighted_loudspeaker_response()
        self.update_statistics()
        self.calculate_filter_spectra(self.mu, self.number_of_eigenvectors)
        self.update_input_blocks(input_A, input_B)
        output_buffers_A, output_buffers_B = self.compute_output_buffers()

        return output_buffers_A, output_buffers_B

    def update_loudspeaker_response_buffers(self, input_A, input_B):
        idx = np.array([i for i in range(self.hop_size, self.block_size)]) # NOTE: could be wrong?
        for m in range(self.number_of_mics):
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
        target_A_to_A_spectra = np.zeros((self.block_size, self.number_of_mics), dtype=complex)
        target_B_to_B_spectra = np.zeros((self.block_size, self.number_of_mics), dtype=complex)
        for m in range(self.number_of_mics):
            target_A_to_A_spectra[:, m] = np.fft.fft(np.multiply(self.window.squeeze(1), self.loudspeaker_target_response_A_to_A_buffer[:, m]), axis=0)
            target_B_to_B_spectra[:, m] = np.fft.fft(np.multiply(self.window.squeeze(1), self.loudspeaker_target_response_B_to_B_buffer[:, m]), axis=0)

        self.update_perceptual_weighting(target_A_to_A_spectra, target_B_to_B_spectra)

        # circular convolution with weighting filter
        target_A_to_A_spectra = np.multiply(target_A_to_A_spectra, self.weighting_spectra_A)
        target_B_to_B_spectra = np.multiply(target_B_to_B_spectra, self.weighting_spectra_B)

        # WOLA reconstruction
        for m in range(self.number_of_mics):
            # zone A
            tmp_old = self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[:, m]
            tmp_new = np.multiply(self.window.squeeze(1), np.fft.ifft(target_A_to_A_spectra[:, m], axis=0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[:, m] = np.pad(tmp_old[self.hop_size: self.block_size], (0, self.hop_size)) + tmp_new

            # Zone B
            tmp_old = self.loudspeaker_weighted_target_response_B_to_B_overlap_buffer[:, m]
            tmp_new = np.multiply(self.window.squeeze(1), np.fft.ifft(target_B_to_B_spectra[:, m], axis=0))
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
        A_to_A_spectra = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics), dtype=complex)
        A_to_B_spectra = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics), dtype=complex)
        B_to_A_spectra = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics), dtype=complex)
        B_to_B_spectra = np.zeros((self.block_size, self.number_of_srcs, self.number_of_mics), dtype=complex)
        for m in range(self.number_of_mics):
            tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_A_to_A_buffer[:, :, m]
            A_to_A_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)
            tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_A_to_B_buffer[:, :, m]
            A_to_B_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)
            tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_B_to_B_buffer[:, :, m]
            B_to_B_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)
            tmp = np.tile(self.window, (1, self.number_of_srcs)) * self.loudspeaker_response_B_to_A_buffer[:, :, m]
            B_to_A_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)

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
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.ifft(A_to_A_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_A_to_A_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal A to zone B
            tmp_old = self.loudspeaker_weighted_response_A_to_B_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.ifft(A_to_B_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_A_to_B_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal B to zone A
            tmp_old = self.loudspeaker_weighted_response_B_to_A_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.ifft(B_to_A_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_B_to_A_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal B to zone B
            tmp_old = self.loudspeaker_weighted_response_B_to_B_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window, (1, self.number_of_srcs)), np.fft.ifft(B_to_B_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_B_to_B_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

        # update weighted_target_response_buffers
        idx = np.array([i for i in range(self.hop_size, self.statistics_buffer_length)]) # NOTE: could be wrong?
        for m in range(self.number_of_mics):
            self.loudspeaker_weighted_response_A_to_A_buffer[:, :, m] = np.concatenate([
                self.loudspeaker_weighted_response_A_to_A_buffer[idx, :, m], 
                self.loudspeaker_weighted_response_A_to_A_overlap_buffer[0:self.hop_size, :, m]])

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
        self.weighting_spectra_A = np.ones((self.block_size, self.number_of_mics)) + 0.0j * np.ones((self.block_size, self.number_of_mics))
        self.weighting_spectra_B = np.ones((self.block_size, self.number_of_mics)) + 0.0j * np.ones((self.block_size, self.number_of_mics))

    def update_statistics(self):
        self.reset_statistics()
        for n in range(self.statistics_buffer_length - self.filter_length):
            idx = n + np.arange(self.filter_length) # NOTE: could be wrong

            self.y_A_to_A = np.zeros((self.number_of_mics, self.number_of_srcs * self.filter_length, 1))
            self.y_A_to_B = np.zeros((self.number_of_mics, self.number_of_srcs * self.filter_length, 1))
            self.y_B_to_A = np.zeros((self.number_of_mics, self.number_of_srcs * self.filter_length, 1))
            self.y_B_to_B = np.zeros((self.number_of_mics, self.number_of_srcs * self.filter_length, 1))

            for m in range(self.number_of_mics):
                self.y_A_to_A[m, :, :] = np.flipud(self.loudspeaker_weighted_response_A_to_A_buffer[idx, :, m]).reshape(-1, 1)
                self.y_A_to_B[m, :, :] = np.flipud(self.loudspeaker_weighted_response_A_to_B_buffer[idx, :, m]).reshape(-1, 1)
                self.y_B_to_A[m, :, :] = np.flipud(self.loudspeaker_weighted_response_B_to_A_buffer[idx, :, m]).reshape(-1, 1)
                self.y_B_to_B[m, :, :] = np.flipud(self.loudspeaker_weighted_response_B_to_B_buffer[idx, :, m]).reshape(-1, 1)

                d_A = np.flipud(self.loudspeaker_weighted_target_response_A_to_A_buffer[idx, m])
                d_B = np.flipud(self.loudspeaker_weighted_target_response_B_to_B_buffer[idx, m])

                self.R_A_to_A = np.add(self.R_A_to_A, np.outer(self.y_A_to_A[m], self.y_A_to_A[m]))
                self.R_A_to_B = np.add(self.R_A_to_B, np.outer(self.y_A_to_B[m], self.y_A_to_B[m]))
                self.R_B_to_A = np.add(self.R_B_to_A, np.outer(self.y_B_to_A[m], self.y_B_to_A[m]))
                self.R_B_to_B = np.add(self.R_B_to_B, np.outer(self.y_B_to_B[m], self.y_B_to_B[m]))

                self.r_A[:,:] = np.add(self.r_A, np.multiply(self.y_A_to_A[m], d_A[0]).reshape(-1, 1))
                self.r_B[:,:] = np.add(self.r_B, np.multiply(self.y_B_to_B[m], d_B[0]).reshape(-1, 1))

    def reset_statistics(self):
        self.R_A_to_A = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.R_A_to_B = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.R_B_to_A = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.R_B_to_B = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.r_A = np.zeros((self.filter_length * self.number_of_srcs, 1))
        self.r_B = np.zeros((self.filter_length * self.number_of_srcs, 1))

    def calculate_filter_spectra(self, mu, number_of_eigenvectors):
        self.U_A, self.lambda_A = jdiag(self.R_A_to_A, self.R_A_to_B)
        self.U_B, self.lambda_B = jdiag(self.R_B_to_B, self.R_B_to_A)

        self.lambda_A = np.diag(self.lambda_A)
        self.lambda_B = np.diag(self.lambda_B)

        self.w_A = np.zeros((self.filter_length * self.number_of_srcs, 1))
        self.w_B = np.zeros((self.filter_length * self.number_of_srcs, 1))
        for i in range(number_of_eigenvectors):
            self.w_A = np.add(self.w_A, np.multiply(np.multiply(self.U_A[:, i].reshape(-1, 1), self.r_A) / (self.lambda_A[i] + mu), self.U_A[:, i].reshape(-1, 1)))
            self.w_B = np.add(self.w_B, np.multiply(np.multiply(self.U_B[:, i].reshape(-1, 1), self.r_B) / (self.lambda_B[i] + mu), self.U_B[:, i].reshape(-1, 1)))

        self.filter_spectra_A = np.fft.fft(np.reshape(self.w_A, (self.filter_length, self.number_of_srcs, 1)), self.block_size, 0).squeeze(2)
        self.filter_spectra_B = np.fft.fft(np.reshape(self.w_B, (self.filter_length, self.number_of_srcs, 1)), self.block_size, 0).squeeze(2)

    def update_input_blocks(self, input_A, input_B):
        self.input_A_block = np.concatenate([self.input_A_block.squeeze(1)[self.hop_size : self.block_size], input_A]).reshape(-1, 1)
        self.input_B_block = np.concatenate([self.input_B_block.squeeze(1)[self.hop_size : self.block_size], input_B]).reshape(-1, 1)

    def compute_output_buffers(self):
        # compute input spectra
        self.input_spectrum_A = np.fft.fft(np.multiply(self.window.squeeze(1), self.input_A_block.squeeze(1)), axis=0).reshape(-1, 1)
        self.input_spectrum_B = np.fft.fft(np.multiply(self.window.squeeze(1), self.input_B_block.squeeze(1)), axis=0).reshape(-1, 1)

        # circular convolution with the filter spectra
        output_spectra_A = np.multiply(np.tile(self.input_spectrum_A, (1, self.number_of_srcs)), self.filter_spectra_A)
        output_spectra_B = np.multiply(np.tile(self.input_spectrum_B, (1, self.number_of_srcs)), self.filter_spectra_B)

        # update the output overlap buffers
        idx = np.arange(self.hop_size, self.block_size)

        self.output_A_overlap_buffer = np.concatenate([
                    self.output_A_overlap_buffer[idx, :], 
                    np.zeros((self.hop_size, self.number_of_srcs))
                ]) 
        tmp = np.fft.ifft(output_spectra_A, self.block_size, axis=0)
        assert np.linalg.norm(np.imag(tmp)) < 1e-8
        tmp = np.real(tmp)
        tmp = np.multiply(tmp, np.tile(self.window, (1, self.number_of_srcs)))
        self.output_A_overlap_buffer += tmp 

        self.output_B_overlap_buffer = np.concatenate([
                    self.output_B_overlap_buffer[idx, :], 
                    np.zeros((self.hop_size, self.number_of_srcs))
                ]) 
        tmp = np.fft.ifft(output_spectra_B, self.block_size, axis=0)
        assert np.linalg.norm(np.imag(tmp)) < 1e-8
        tmp = np.real(tmp)
        tmp = np.multiply(tmp, np.tile(self.window, (1, self.number_of_srcs)))
        self.output_B_overlap_buffer += tmp 

        # extract samples for the output buffers
        output_buffer_A = self.output_A_overlap_buffer[:self.hop_size, :]
        output_buffer_B = self.output_B_overlap_buffer[:self.hop_size, :]

        return output_buffer_A, output_buffer_B

if __name__ == "__main__":
    import pytest
    reltol = 1e-9
    abstol = 1e-20

    np.seterr(all='raise')

    print(f"Loading test.mat...")
    test = sp.io.loadmat("test.mat")
    print(np.any(test["wAb"] > 0))
    print(f"Loading test.mat OK")

    print(f"Creating AP-VAST object...")
    ap = apvast(
        block_size=test["blockSize"][0][0],
        rir_A=test["rirA"],
        rir_B=test["rirB"],
        filter_length=test["filterLength"][0][0],
        modeling_delay=test["modelingDelay"][0][0],
        reference_index_A=test["referenceIndexA"][0][0] - 1, # python vs matlab indexing, it is what it is
        reference_index_B=test["referenceIndexB"][0][0] - 1, # python vs matlab indexing, it is what it is
        number_of_eigenvectors=test["numberOfEigenVectors"][0][0],
        mu=test["mu"][0][0],
        statistics_buffer_length=test["statisticsBufferLength"][0][0],
        hop_size=test["hopSize"][0][0],
    )
    print(f"Creating AP-VAST object OK")

    print(f"Asserting object the same...")
    np.testing.assert_allclose(ap.block_size, test["before_m_blockSize"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.filter_length, test["before_m_filterLength"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.hop_size, test["before_m_hopSize"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.input_A_block, test["before_m_inputABlock"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.input_B_block, test["before_m_inputBBlock"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_A_to_A_buffer, test["before_m_loudspeakerResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_A_to_B_buffer, test["before_m_loudspeakerResponseAtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_B_to_A_buffer, test["before_m_loudspeakerResponseBtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_B_to_B_buffer, test["before_m_loudspeakerResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_target_response_A_to_A_buffer, test["before_m_loudspeakerTargetResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_target_response_B_to_B_buffer, test["before_m_loudspeakerTargetResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_A_buffer, test["before_m_loudspeakerWeightedResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_A_overlap_buffer, test["before_m_loudspeakerWeightedResponseAtoAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_B_buffer, test["before_m_loudspeakerWeightedResponseAtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_B_overlap_buffer, test["before_m_loudspeakerWeightedResponseAtoBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_A_buffer, test["before_m_loudspeakerWeightedResponseBtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_A_overlap_buffer, test["before_m_loudspeakerWeightedResponseBtoAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_B_buffer, test["before_m_loudspeakerWeightedResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_B_overlap_buffer, test["before_m_loudspeakerWeightedResponseBtoBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_A_to_A_buffer, test["before_m_loudspeakerWeightedTargetResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_A_to_A_overlap_buffer, test["before_m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_B_to_B_buffer, test["before_m_loudspeakerWeightedTargetResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_B_to_B_overlap_buffer, test["before_m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.mu, test["before_m_mu"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.number_of_eigenvectors, test["before_m_numberOfEigenvectors"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.number_of_mics, test["before_m_numberOfMics"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.number_of_srcs, test["before_m_numberOfSrcs"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.output_A_overlap_buffer, test["before_m_outputAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.output_B_overlap_buffer, test["before_m_outputBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_A, test["before_m_rirA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_A_to_A_state, test["before_m_rirAtoAState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_A_to_B_state, test["before_m_rirAtoBState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_B, test["before_m_rirB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_B_to_A_state, test["before_m_rirBtoAState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_B_to_B_state, test["before_m_rirBtoBState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_length, test["before_m_rirLength"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.statistics_buffer_length, test["before_m_statisticsBufferLength"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_A, test["before_m_targetRirA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_A_to_A_state, test["before_m_targetRirAtoAState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_B, test["before_m_targetRirB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_B_to_B_state, test["before_m_targetRirBtoBState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.window, test["before_m_window"], rtol=reltol, atol=abstol)
    print(f"Asserting object the same OK")

    print(f"Running...")
    iAb = test["iAb"]
    iBb = test["iBb"]
    oAb = np.zeros_like(test["oAb"])
    oBb = np.zeros_like(test["oAb"])
    for i in range(iAb.shape[0]):
        oAb[i,:,:], oBb[i,:,:] = ap.process_input_buffers(iAb[i,:], iBb[i,:])
        print(f'reference w_A, first 5 samples first source:\n{test["wAb"][i,0:5]}')
        print(f'reference w_B, first 5 samples first source:\n{test["wBb"][i,0:5]}')
        print(f'obtained w_A, first 5 samples first source:\n{ap.w_A[0:5,0]}')
        print(f'obtained w_B, first 5 samples first source:\n{ap.w_B[0:5,0]}')
    print(f"Running OK")

    import matplotlib.pyplot as plt
    print(f"Asserting object the same...")
    np.testing.assert_allclose(ap.block_size, test["after_m_blockSize"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.filter_length, test["after_m_filterLength"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.hop_size, test["after_m_hopSize"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_target_response_A_to_A_buffer, test["after_m_loudspeakerTargetResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_target_response_B_to_B_buffer, test["after_m_loudspeakerTargetResponseBtoBBuffer"], rtol=reltol, atol=abstol)

    for i in range(test["after_m_loudspeakerResponseAtoABuffer"].shape[0]):
        print(ap.loudspeaker_response_A_to_A_buffer[i].shape)
        diff = np.linalg.norm(ap.loudspeaker_response_A_to_A_buffer[i] - test["after_m_loudspeakerResponseAtoABuffer"][i])
        if diff > 1e-15:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(10 * np.log10(np.abs(ap.loudspeaker_response_A_to_A_buffer[i]) + 1e-25))
            axs[1].imshow(10 * np.log10(np.abs(test["after_m_loudspeakerResponseAtoABuffer"][i]) + 1e-25))
            plt.show()
        else:
            print(f"skipping {i}, diff {diff}...")

    # for i in range(test["after_m_loudspeakerResponseAtoABuffer"].shape[0]):
    #     for j in range(test["after_m_loudspeakerResponseAtoABuffer"].shape[1]):
    #         for k in range(test["after_m_loudspeakerResponseAtoABuffer"].shape[2]):
    #             epstol = 1e-30
    #             print(test["after_m_loudspeakerResponseAtoABuffer"][i,j,k])
    #             print(ap.loudspeaker_response_A_to_A_buffer[i,j,k])
    #             assert abs(test["after_m_loudspeakerResponseAtoABuffer"][i,j,k] - ap.loudspeaker_response_A_to_A_buffer[i,j,k]) < abstol
    #             assert abs(test["after_m_loudspeakerResponseAtoABuffer"][i,j,k] - ap.loudspeaker_response_A_to_A_buffer[i,j,k]) / abs(test["after_m_loudspeakerResponseAtoABuffer"][i,j,k] + epstol) < reltol

    np.testing.assert_allclose(ap.loudspeaker_response_A_to_A_buffer, test["after_m_loudspeakerResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_A_to_B_buffer, test["after_m_loudspeakerResponseAtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_B_to_A_buffer, test["after_m_loudspeakerResponseBtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_response_B_to_B_buffer, test["after_m_loudspeakerResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_A_overlap_buffer, test["after_m_loudspeakerWeightedResponseAtoAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_A_buffer, test["after_m_loudspeakerWeightedResponseAtoABuffer"], rtol=reltol, atol=abstol) 
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_B_overlap_buffer, test["after_m_loudspeakerWeightedResponseAtoBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_A_to_B_buffer, test["after_m_loudspeakerWeightedResponseAtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_A_overlap_buffer, test["after_m_loudspeakerWeightedResponseBtoAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_A_buffer, test["after_m_loudspeakerWeightedResponseBtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_B_overlap_buffer, test["after_m_loudspeakerWeightedResponseBtoBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_response_B_to_B_buffer, test["after_m_loudspeakerWeightedResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_A_to_A_buffer, test["after_m_loudspeakerWeightedTargetResponseAtoABuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_A_to_A_overlap_buffer, test["after_m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_B_to_B_buffer, test["after_m_loudspeakerWeightedTargetResponseBtoBBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.loudspeaker_weighted_target_response_B_to_B_overlap_buffer, test["after_m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.mu, test["after_m_mu"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.number_of_eigenvectors, test["after_m_numberOfEigenvectors"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.number_of_mics, test["after_m_numberOfMics"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.number_of_srcs, test["after_m_numberOfSrcs"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.input_A_block, test["after_m_inputABlock"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.input_B_block, test["after_m_inputBBlock"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.weighting_spectra_A, test["after_m_weightingSpectraA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.weighting_spectra_B, test["after_m_weightingSpectraB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.window, test["after_m_window"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_A, test["after_m_rirA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_A_to_A_state, test["after_m_rirAtoAState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_A_to_B_state, test["after_m_rirAtoBState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_B, test["after_m_rirB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_B_to_A_state, test["after_m_rirBtoAState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_B_to_B_state, test["after_m_rirBtoBState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.rir_length, test["after_m_rirLength"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.statistics_buffer_length, test["after_m_statisticsBufferLength"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_A, test["after_m_targetRirA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_A_to_A_state, test["after_m_targetRirAtoAState"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_B, test["after_m_targetRirB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.target_rir_B_to_B_state, test["after_m_targetRirBtoBState"], rtol=reltol, atol=abstol)

    # different?
    np.testing.assert_allclose(ap.R_A_to_A, test["after_m_RAtoA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.R_A_to_B, test["after_m_RAtoB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.R_B_to_A, test["after_m_RBtoA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.R_B_to_B, test["after_m_RBtoB"], rtol=reltol, atol=abstol)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns
    # axs[0].imshow(10 * np.log10(np.abs(ap.R_A_to_A)))
    # axs[1].imshow(10 * np.log10(np.abs(test["after_m_RAtoA"])))
    # plt.show()

    np.testing.assert_allclose(ap.r_A, test["after_m_rA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.r_B, test["after_m_rB"], rtol=reltol, atol=abstol)

    # different:
    np.testing.assert_allclose(ap.U_A, test["after_m_UA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.U_B, test["after_m_UB"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.lambda_A, test["after_m_lambdaA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.lambda_B, test["after_m_lambdaB"], rtol=reltol, atol=abstol)

    # different:
    np.testing.assert_allclose(ap.w_A, test["after_m_wA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.w_B, test["after_m_wB"], rtol=reltol, atol=abstol)

    # different:
    np.testing.assert_allclose(ap.filter_spectra_A, test["after_m_filterSpectraA"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.filter_spectra_B, test["after_m_filterSpectraB"], rtol=reltol, atol=abstol)

    np.testing.assert_allclose(ap.output_A_overlap_buffer, test["after_m_outputAOverlapBuffer"], rtol=reltol, atol=abstol)
    np.testing.assert_allclose(ap.output_B_overlap_buffer, test["after_m_outputBOverlapBuffer"], rtol=reltol, atol=abstol)
    print(f"Asserting object the same OK")

    print(f"Asserting output the same...")
    print(f"Asserting object the same OK")

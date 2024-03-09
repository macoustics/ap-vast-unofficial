import numpy as np
import scipy as sp

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
        self.window = np.sin(np.pi / self.block_size * np.arange(self.block_size))
        self.input_A_block = np.zeros((self.block_size))
        self.input_B_block = np.zeros((self.block_size))
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
            target_A_to_A_spectra[:, m] = np.fft.fft(np.multiply(self.window, self.loudspeaker_target_response_A_to_A_buffer[:, m]), axis=0)
            target_B_to_B_spectra[:, m] = np.fft.fft(np.multiply(self.window, self.loudspeaker_target_response_B_to_B_buffer[:, m]), axis=0)

        self.update_perceptual_weighting(target_A_to_A_spectra, target_B_to_B_spectra)

        # circular convolution with weighting filter
        target_A_to_A_spectra = np.multiply(target_A_to_A_spectra, self.weighting_spectra_A)
        target_B_to_B_spectra = np.multiply(target_B_to_B_spectra, self.weighting_spectra_B)

        # WOLA reconstruction
        for m in range(self.number_of_mics):
            # zone A
            tmp_old = self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[:, m]
            tmp_new = np.multiply(self.window, np.fft.ifft(target_A_to_A_spectra[:, m], axis=0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_target_response_A_to_A_overlap_buffer[:, m] = np.pad(tmp_old[self.hop_size: self.block_size], (0, self.hop_size)) + tmp_new

            # Zone B
            tmp_old = self.loudspeaker_weighted_target_response_B_to_B_overlap_buffer[:, m]
            tmp_new = np.multiply(self.window, np.fft.ifft(target_B_to_B_spectra[:, m], axis=0))
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
            tmp = np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)) * self.loudspeaker_response_A_to_A_buffer[:, :, m]
            A_to_A_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)
            tmp = np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)) * self.loudspeaker_response_A_to_B_buffer[:, :, m]
            A_to_B_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)
            tmp = np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)) * self.loudspeaker_response_B_to_B_buffer[:, :, m]
            B_to_B_spectra[:, :, m] = np.fft.fft(tmp, self.block_size, 0)
            tmp = np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)) * self.loudspeaker_response_B_to_A_buffer[:, :, m]
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
            tmp_new = np.multiply(np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)), np.fft.ifft(A_to_A_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_A_to_A_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal A to zone B
            tmp_old = self.loudspeaker_weighted_response_A_to_B_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)), np.fft.ifft(A_to_B_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_A_to_B_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal B to zone A
            tmp_old = self.loudspeaker_weighted_response_B_to_A_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)), np.fft.ifft(B_to_A_spectra[:, :, m], self.block_size, 0))
            assert np.linalg.norm(np.imag(tmp_new)) < 1e-8
            tmp_new = np.real(tmp_new)
            self.loudspeaker_weighted_response_B_to_A_overlap_buffer[:, :, m] = np.concatenate([tmp_old[idx, :], np.zeros((self.hop_size, self.number_of_srcs))]) + tmp_new

            # signal B to zone B
            tmp_old = self.loudspeaker_weighted_response_B_to_B_overlap_buffer[:, :, m]
            tmp_new = np.multiply(np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)), np.fft.ifft(B_to_B_spectra[:, :, m], self.block_size, 0))
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
            for m in range(self.number_of_mics):
                y_A_to_A = np.flipud(self.loudspeaker_weighted_response_A_to_A_buffer[idx, :, m]).reshape(-1, 1)
                y_A_to_B = np.flipud(self.loudspeaker_weighted_response_A_to_B_buffer[idx, :, m]).reshape(-1, 1)
                y_B_to_A = np.flipud(self.loudspeaker_weighted_response_B_to_A_buffer[idx, :, m]).reshape(-1, 1)
                y_B_to_B = np.flipud(self.loudspeaker_weighted_response_B_to_B_buffer[idx, :, m]).reshape(-1, 1)

                d_A = np.flipud(self.loudspeaker_weighted_target_response_A_to_A_buffer[idx, m])
                d_B = np.flipud(self.loudspeaker_weighted_target_response_B_to_B_buffer[idx, m])

                self.R_A_to_A = np.add(self.R_A_to_A, np.outer(y_A_to_A, y_A_to_A))
                self.R_A_to_B = np.add(self.R_A_to_B, np.outer(y_A_to_B, y_A_to_B))
                self.R_B_to_A = np.add(self.R_B_to_A, np.outer(y_B_to_A, y_B_to_A))
                self.R_B_to_B = np.add(self.R_B_to_B, np.outer(y_B_to_B, y_B_to_B))

                self.r_A[:,:] = np.add(self.r_A, np.multiply(y_A_to_A, d_A[0]).reshape(-1, 1))
                self.r_B[:,:] = np.add(self.r_B, np.multiply(y_B_to_B, d_B[0]).reshape(-1, 1))
            print(self.R_A_to_A)

    def reset_statistics(self):
        self.R_A_to_A = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.R_A_to_B = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.R_B_to_A = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.R_B_to_B = np.zeros((self.filter_length * self.number_of_srcs, self.filter_length * self.number_of_srcs))
        self.r_A = np.zeros((self.filter_length * self.number_of_srcs, 1))
        self.r_B = np.zeros((self.filter_length * self.number_of_srcs, 1))

    def calculate_filter_spectra(self, mu, number_of_eigenvectors):
        U_A, lambda_A = jdiag(self.R_A_to_A, self.R_A_to_B)
        U_B, lambda_B = jdiag(self.R_B_to_B, self.R_B_to_A)

        lambda_A = np.diag(lambda_A)
        lambda_B = np.diag(lambda_B)

        w_A = np.zeros((self.filter_length * self.number_of_srcs, 1))
        w_B = np.zeros((self.filter_length * self.number_of_srcs, 1))
        for i in range(number_of_eigenvectors):
            w_A = np.add(w_A, np.multiply(np.multiply(U_A[:, i].reshape(-1, 1), self.r_A) / (lambda_A[i] + mu), U_A[:, i].reshape(-1, 1)))
            w_B = np.add(w_B, np.multiply(np.multiply(U_B[:, i].reshape(-1, 1), self.r_B) / (lambda_B[i] + mu), U_B[:, i].reshape(-1, 1)))

        self.filter_spectra_A = np.fft.fft(np.reshape(w_A, (self.filter_length, self.number_of_srcs, 1)), self.block_size, 0)
        self.filter_spectra_B = np.fft.fft(np.reshape(w_B, (self.filter_length, self.number_of_srcs, 1)), self.block_size, 0)

    def update_input_blocks(self, input_A, input_B):
        self.input_A_block = np.concatenate([self.input_A_block[self.hop_size : self.block_size], input_A])
        self.input_B_block = np.concatenate([self.input_B_block[self.hop_size : self.block_size], input_B])

    def compute_output_buffers(self):
        # compute input spectra
        input_spectrum_A = np.fft.fft(np.multiply(self.window, self.input_A_block), axis=0).reshape(-1, 1)
        input_spectrum_B = np.fft.fft(np.multiply(self.window, self.input_B_block), axis=0).reshape(-1, 1)

        # circular convolution with the filter spectra
        output_spectra_A = np.multiply(np.tile(input_spectrum_A, (1, self.number_of_srcs))[:,:,np.newaxis], self.filter_spectra_A)
        output_spectra_B = np.multiply(np.tile(input_spectrum_B, (1, self.number_of_srcs))[:,:,np.newaxis], self.filter_spectra_B)

        # update the output overlap buffers
        idx = np.arange(self.hop_size, self.block_size)

        self.output_A_overlap_buffer = np.concatenate([
                    self.output_A_overlap_buffer[idx, :], 
                    np.zeros((self.hop_size, self.number_of_srcs))
                ]) 
        tmp = np.fft.ifft(output_spectra_A, self.block_size, axis=0).squeeze(-1)
        assert np.linalg.norm(np.imag(tmp)) < 1e-8
        tmp = np.real(tmp)
        tmp = np.multiply(tmp, np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)))
        self.output_A_overlap_buffer += tmp 

        self.output_B_overlap_buffer = np.concatenate([
                    self.output_B_overlap_buffer[idx, :], 
                    np.zeros((self.hop_size, self.number_of_srcs))
                ]) 
        tmp = np.fft.ifft(output_spectra_B, self.block_size, axis=0).squeeze(-1)
        assert np.linalg.norm(np.imag(tmp)) < 1e-8
        tmp = np.real(tmp)
        tmp = np.multiply(tmp, np.tile(self.window.reshape(-1, 1), (1, self.number_of_srcs)))
        self.output_B_overlap_buffer += tmp 

        # extract samples for the output buffers
        output_buffer_A = self.output_A_overlap_buffer[:self.hop_size, :]
        output_buffer_B = self.output_B_overlap_buffer[:self.hop_size, :]

        return output_buffer_A, output_buffer_B

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.seterr(all='raise')

    exit(1)
    rirs = sp.io.loadmat("rirs.mat")
    ap = apvast(
        block_size=2 * mat["rirA"].shape[0],
        rir_A=rirs["rirA"],
        rir_B=rirs["rirB"],
        filter_length=100,
        modeling_delay=20,
        reference_index_A=7,
        reference_index_B=7,
        number_of_eigenvectors=50,
        mu=1.0,
        statistics_buffer_length=1000,
        hop_size=rirs["rirA"].shape[0],
    )

    iA = np.random.randn(rirs["rirA"].shape[0])
    iB = np.random.randn(rirs["rirA"].shape[0])
    oA, oB = ap.process_input_buffers(iA, iB)

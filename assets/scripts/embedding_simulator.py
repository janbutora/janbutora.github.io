import numpy as np
import os


"""
Tools for steganographic algorithms
"""

class Embedding_simulator():
    @staticmethod
    def binary_entropy(pPM1):
        p0 = 1 - pPM1
        p0[p0 <= 0] = 1
        pPM1[pPM1==0]=1
        P = np.hstack((p0.flatten(), pPM1.flatten()))
        H = -((P) * np.log2(P))
        Ht = np.nansum(H)
        return Ht

    @staticmethod
    def ternary_entropy(pP1, pM1):
        p0 = 1 - pP1 - pM1
        p0[p0 <= 0] = 1
        pP1[pP1==0]=1
        pM1[pM1==0]=1
        P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
        H = -((P) * np.log2(P))
        Ht = np.nansum(H)
        return Ht

    @staticmethod
    def calc_lambda_binary(rhoPM1, message_length, n):
        l3 = 1000 # just an initial value
        m3 = message_length + 1 # to enter at least one time in the loop, just an initial value
                                # m3 is the total entropy
        iterations = 0 # iterations counter
        while m3 > message_length:
            """
            This loop returns the biggest l3 such that total entropy (m3) <= message_length
            Stop when total entropy < message_length
            """
            l3 *= 2
            pPM1 = np.exp(-l3 * rhoPM1) / (1 + np.exp(-l3 * rhoPM1))
            # Total entropy
            m3 = Embedding_simulator.binary_entropy(pPM1)
            iterations += 1
            if iterations > 10:
                """
                Probably unbounded => it seems that we can't find beta such that
                message_length will be smaller than requested. Binary search
                doesn't work here
                """
                lbd = l3
                return lbd


        l1 = 0.0 # just an initial value
        m1 = n # just an initial value
        lbd = 0.0

        alpha = message_length / n # embedding rate
        # Limit search to 30 iterations
        # Require that relative payload embedded is roughly within
        # 1/1000 of the required relative payload
        while (m1 - m3) / n > alpha / 1000 and iterations < 30:
            lbd = l1 + (l3 - l1) / 2 # dichotomy
            pPM1 = np.exp(-lbd * rhoPM1) / (1 + np.exp(-lbd * rhoPM1))
            m2 = Embedding_simulator.binary_entropy(pPM1) # total entropy new calculation
            if m2 < message_length: # classical binary search
                l3 = lbd
                m3 = m2
            else:
                l1 = lbd
                m1 = m2
            iterations += 1 # for monitoring the number of iterations
        return lbd

    @staticmethod
    def calc_lambda(rhoP1, rhoM1, message_length, n):
        l3 = 1000 # just an initial value
        m3 = message_length + 1 # to enter at least one time in the loop, just an initial value
                                # m3 is the total entropy
        iterations = 0 # iterations counter
        while m3 > message_length:
            """
            This loop returns the biggest l3 such that total entropy (m3) <= message_length
            Stop when total entropy < message_length
            """
            l3 *= 2
            pP1 = np.exp(-l3 * rhoP1) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
            pM1 = np.exp(-l3 * rhoM1) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
            # Total entropy
            m3 = Embedding_simulator.ternary_entropy(pP1, pM1)
            iterations += 1
            if iterations > 10:
                """
                Probably unbounded => it seems that we can't find beta such that
                message_length will be smaller than requested. Ternary search
                doesn't work here
                """
                lbd = l3
                return lbd


        l1 = 0.0 # just an initial value
        m1 = n # just an initial value
        lbd = 0.0

        alpha = message_length / n # embedding rate
        # Limit search to 30 iterations
        # Require that relative payload embedded is roughly within
        # 1/1000 of the required relative payload
        while (m1 - m3) / n > alpha / 1000 and iterations < 30:
            lbd = l1 + (l3 - l1) / 2 # dichotomy
            pP1 = np.exp(-lbd * rhoP1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
            pM1 = np.exp(-lbd * rhoM1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
            m2 = Embedding_simulator.ternary_entropy(pP1, pM1) # total entropy new calculation
            if m2 < message_length: # classical ternary search
                l3 = lbd
                m3 = m2
            else:
                l1 = lbd
                m1 = m2
            iterations += 1 # for monitoring the number of iterations
        return lbd

    @staticmethod
    def calc_lambda_FI(payload, solver, kargs):
        [L,R] = [1e-6, 1e-2]
        fL = Embedding_simulator.ternary_entropy(*solver(L, *kargs)) - payload
        fR = Embedding_simulator.ternary_entropy(*solver(R, *kargs)) - payload
        #pdb.set_trace()
        max_iter = 20
        i = 0
        while (fL*fR > 0) and (i<max_iter):
            i += 1
            if fL < 0:
                L = R
                R *= 2
                fR = Embedding_simulator.ternary_entropy(*solver(R, *kargs)) - payload
            else:
                R = L
                L /= 2
                fL = Embedding_simulator.ternary_entropy(*solver(L, *kargs)) - payload

        i, fM, TM = 0, 1, np.zeros([max_iter,2])
        while (np.abs(fM) >= 1e0) and (i < max_iter):
            M = (L+R)/2
            fM = Embedding_simulator.ternary_entropy(*solver(M, *kargs)) - payload
            if fL*fM < 0:
                R = M
                fR = fM
            else:
                L = M
                fL = fM
            TM[i,:] = [fM, M]
            i += 1

        if i == max_iter:
            M = TM[np.argmin(np.abs(TM[:i,0])),1]

        pP1, pM1 = solver(M, *kargs)

        return pP1, pM1, M

    @staticmethod
    def calc_lambda_FI_binary(FI, payload):
        L, R = 5e-2, 5e1
        ixlnx2 = np.load('ixlnx2.npy')
        fL = Embedding_simulator.binary_entropy(1/invxlnx2_fast(L*FI, ixlnx2)) - payload
        fR = Embedding_simulator.binary_entropy(1/invxlnx2_fast(R*FI, ixlnx2)) - payload
        max_iter = 80
        i = 0
        while (fL*fR > 0) and (i<max_iter):
            i += 1
            if fL > 0:
                R *= 2
                fR = Embedding_simulator.binary_entropy(1/invxlnx2_fast(R*FI, ixlnx2)) - payload
            else:
                L /= 2
                fL = Embedding_simulator.binary_entropy(1/invxlnx2_fast(L*FI, ixlnx2)) - payload

        i, fM, TM = 0, 1, np.zeros([max_iter,2])
        while (np.abs(fM) > 1e-2) and (i < max_iter):
            M = (L+R)/2
            fM = Embedding_simulator.binary_entropy(1/invxlnx2_fast(M*FI, ixlnx2)) - payload
            if fL*fM < 0:
                R = M
                fR = fM
            else:
                L = M
                fl = fM
            TM[i,:] = [fM, M]
            i += 1

        if i == max_iter:
            M = TM[np.argmin(np.abs(TM[:i,0])),1]

        beta = 1/invxlnx2_fast(M*FI, ixlnx2)
        return beta

    @staticmethod
    def invxlnx2_fast(y,f):
        i_large = y>=1000
        i_small = y<1000
        iyL = (np.floor(y[i_small]/0.01)).astype(np.int32)
        iyR = iyL + 1
        iyR[iyR>=100000] = 100000-1

        x = np.zeros(y.shape)
        x[i_small] = f[iyL] + (y[i_small]-(iyL)*0.01)*(f[iyR]-f[iyL])

        z = y[i_large]/np.log(y[i_large]-1)
        for j in range(20):
            z = y[i_large]/np.log(z-1)

        x[i_large] = z
        return x

    @staticmethod
    def compute_proba_binary(rhoPM1, message_length, n):
        """
        Embedding simulator simulates the embedding made by the best possible
        binary coding method (it embeds on the entropy bound). This can be
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC)
        that are asymptotically approaching the bound
        """
        lbd = Embedding_simulator.calc_lambda_binary(rhoPM1, message_length, n)
        p_change_PM1 = np.exp(-lbd * rhoPM1) / (1 + np.exp(-lbd * rhoPM1))
        return p_change_PM1

    @staticmethod
    def compute_proba(rhoP1, rhoM1, message_length, n):
        """
        Embedding simulator simulates the embedding made by the best possible
        ternary coding method (it embeds on the entropy bound). This can be
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC)
        that are asymptotically approaching the bound
        """
        lbd = Embedding_simulator.calc_lambda(rhoP1, rhoM1, message_length, n)
        p_change_P1 = np.exp(-lbd * rhoP1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
        p_change_M1 = np.exp(-lbd * rhoM1) / (1 + np.exp(-lbd * rhoP1) + np.exp(-lbd * rhoM1))
        return p_change_P1, p_change_M1

    @staticmethod
    def process_binary(cover, rhoPM1, message_length):
        """
        Embedding simulator simulates the embedding made by the best possible
        binary coding method (it embeds on the entropy bound). This can be
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC)
        that are asymptotically approaching the bound
        """

        np.random.seed(np.fromstring(os.urandom(32),dtype=np.uint32)[0])

        n = cover.size
        p_change_PM1 = Embedding_simulator.compute_proba_binary(rhoPM1, message_length, n)

        randChange = np.random.random_sample((cover.shape[0], cover.shape[1]))
        y = np.copy(cover)
        y[randChange < p_change_PM1] = y[randChange < p_change_PM1] - 1 + 2 * np.random.randint(2)
        return y, p_change_PM1

    @staticmethod
    def process(cover, p_change_P1, p_change_M1):
        """
        Embedding simulator simulates the embedding made by the best possible
        ternary coding method (it embeds on the entropy bound). This can be
        achieved in practice using Multi-layered syndrome-trellis codes (ML STC)
        that are asymptotically approaching the bound
        """

        np.random.seed(np.fromstring(os.urandom(32),dtype=np.uint32)[0])
        if cover.ndim>2:
            randChange = np.random.random_sample((cover.shape[0], cover.shape[1]))
        else:
            randChange = np.random.random_sample(cover.shape)

        y = np.copy(cover)
        y[randChange < p_change_P1] = y[randChange < p_change_P1] + 1
        y[np.logical_and(randChange >= p_change_P1, randChange < (p_change_P1 + p_change_M1))] = y[np.logical_and(randChange >= p_change_P1, randChange < (p_change_P1 + p_change_M1))] - 1
        return y

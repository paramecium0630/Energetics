module theory_mod
    use precision_mod
    implicit none

contains

    logical function dummy_select(wr, wi)
        real(dp), intent(in) :: wr, wi

        dummy_select = .false.

    end function dummy_select

    subroutine solve_lyapunov(A, C, X, max_real_part)
    ! solve AX + XA^T = C
        integer :: n, lwork
        integer :: info, sdim
        real(dp), intent(in) :: A(:, :)
        real(dp), intent(in) :: C(:, :)
        real(dp), intent(out) :: X(:, :), max_real_part
        
        real(dp) :: scale

        real(dp), allocatable :: T(:,:)
        real(dp), allocatable :: U(:,:)
        real(dp), allocatable :: Y(:,:)
        real(dp), allocatable :: F(:,:)
        real(dp), allocatable :: tmp(:,:)
        real(dp), allocatable :: wr(:), wi(:)
        real(dp), allocatable :: work(:)
        logical, allocatable :: bwork(:)

        n = size(A, 1) ! size of A

        if (size(A,2) /= n) error stop "A must be square"
        if (size(C,1) /= n .or. size(C,2) /= n) &
            error stop "C has wrong shape"
        if (size(X,1) /= n .or. size(X,2) /= n) &
            error stop "X has wrong shape"

        allocate(T(n,n))
        allocate(U(n,n))
        allocate(F(n,n))
        allocate(Y(n,n))
        allocate(tmp(n,n))
        allocate(wr(n), wi(n))
        allocate(bwork(n))

        ! DGEES overwrites input matrix
        T = A
        
        ! -------------------------------------------------
        ! 1. Workspace query for DGEES
        ! -------------------------------------------------

        lwork = -1
        allocate(work(1))

        call dgees('V', 'N', dummy_select, n, T, n, &
                   sdim, wr, wi, U, n, work, lwork, bwork, info)

        if (info /= 0) then
            error stop "DGEES workspace query failed"
        end if

        lwork = int(work(1)) ! workspace test

        deallocate(work)
        allocate(work(lwork))

        ! -------------------------------------------------
        ! 2. Real Schur decomposition
        !       A = U T U^T
        ! -------------------------------------------------

        T = A

        call dgees('V', 'N', dummy_select, n, T, n, &
                   sdim, wr, wi, U, n, work, lwork, bwork, info)

        if (info /= 0) then
            error stop "DGEES failed"
        end if

        ! 穩定性檢查：continuous-time system
        max_real_part = maxval(wr)

        if (max_real_part >= 0.0_dp) then
            print *, "max Re(lambda(Q)) =", max_real_part
            error stop "Q is not Hurwitz stable"
        end if

        ! -------------------------------------------------
        ! 3. F = U^T C U
        ! -------------------------------------------------

        tmp = matmul(transpose(U), C)
        F   = matmul(tmp, U)

        ! -------------------------------------------------
        ! 4. Solve
        !       T Y + Y T^T = scale * F
        ! DTRSYL will overwrite its RHS,
        ! so copy F into Y
        ! -------------------------------------------------

        Y = F

        call dtrsyl('N', 'T', 1, n, n, &
                    T, n, T, n, Y, n, scale, info)

        if (info < 0) then
            error stop "DTRSYL: illegal argument"
        else if (info == 1) then
            print *, "Warning: Lyapunov equation nearly singular."
        end if

        ! DTRSYL actually solves:
        !
        ! T Y + Y T^T = scale * F
        !
        ! therefore divide by scale to recover
        ! T Y + Y T^T = F

        Y = Y / scale

        ! -------------------------------------------------
        ! 5. Transform back
        !       X = U Y U^T
        ! -------------------------------------------------

        tmp = matmul(U, Y)
        X   = matmul(tmp, transpose(U))

    end subroutine solve_lyapunov

    subroutine analytic_result(Q, noise, K, tau, delta, alpha, Ktau)
        use stdlib_linalg, only : expm
        ! analytic results of delta, K, Ktau
        integer :: n
        real(dp), intent(in) :: Q(:, :), noise(:, :), K(:, :), tau
        real(dp), allocatable, intent(out) :: delta(:, :), Ktau(:, :), alpha(:, :)
        real(dp), allocatable :: expmatrix(:,:)

        n = size(Q, 1)

        allocate(delta(n, n), alpha(n, n), Ktau(n, n), expmatrix(n, n))

        delta = matmul(Q, noise) - matmul(noise, transpose(Q))

        alpha = matmul(K, transpose(Q)) - matmul(Q, K)

        ! expmatrix = expm(Q*tau)

        ! Ktau = matmul(expmatrix, K)

    end subroutine analytic_result

    subroutine compute_energetics_theory(Q, noise, alpha, heat_rate, work_rate, &
                                     internal_rate, entropy_rate)
        real(dp), intent(in) :: Q(:,:), noise(:,:), alpha(:,:)
        real(dp), allocatable, intent(out) :: heat_rate(:), work_rate(:)
        real(dp), allocatable, intent(out) :: internal_rate(:), entropy_rate(:)
        integer :: n, i
        real(dp) :: diag_Qalpha, diag_Qtalpha

        n = size(Q, 1)
        
        if (n <= 0) error stop "Theory size must be positive"
        if (size(Q,2) /= n) error stop "Q must be square"

        if (size(alpha,1) /= n .or. size(alpha,2) /= n) then
        error stop "Q and alpha size mismatch"
        end if

        if (size(noise,1) /= n .or. size(noise,2) /= n) then
        error stop "Q and noise size mismatch"
        end if

        allocate(heat_rate(n), work_rate(n))
        allocate(internal_rate(n), entropy_rate(n))

        do i = 1, n
            if (noise(i,i) <= 0.0_dp) then
            error stop "Noise diagonal must be positive"
            end if

            ! (Q alpha)_ii
            diag_Qalpha = dot_product(Q(i,:), alpha(:,i))

            ! (Q^T alpha)_ii
            diag_Qtalpha = dot_product(Q(:,i), alpha(:,i))

            heat_rate(i) = -0.5_dp * diag_Qalpha

            ! A = (Q - Q^T)/2
            work_rate(i) = -0.25_dp * &
            (diag_Qalpha - diag_Qtalpha)

            ! S = (Q + Q^T)/2
            internal_rate(i) = -0.25_dp * &
            (diag_Qalpha + diag_Qtalpha)

            entropy_rate(i) = diag_Qalpha / noise(i,i)
        end do

    end subroutine compute_energetics_theory

    subroutine simulate_alpha(Ktau, tau, alpha_sim)
        integer :: n
        real(dp) :: tau
        real(dp), intent(in) :: Ktau(:, :)
        real(dp), allocatable, intent(out) :: alpha_sim(:, :)

        n = size(Ktau, 1)

        allocate(alpha_sim(n, n))

        alpha_sim = (transpose(Ktau) - Ktau) / tau

    end subroutine simulate_alpha    

end module theory_mod
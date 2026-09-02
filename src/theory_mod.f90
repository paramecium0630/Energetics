module theory_mod
    use precision_mod
    implicit none

contains

    logical function is_upper_triangular(A, tol)
        real(dp), intent(in) :: A(:, :)
        real(dp), intent(in) :: tol
        integer :: n, i

        n = size(A, 1)

        if (size(A, 2) /= n) error stop "A must be square"
        if (tol < 0.0_dp) error stop "tol must be non-negative"

        is_upper_triangular = .true.

        do i = 2, n
        ! 第 i 欄、對角線左側：下三角元素
            if (any(abs(A(i, 1:i-1)) > tol)) then
                is_upper_triangular = .false.
                return
            end if
        end do

    end function is_upper_triangular

    logical function is_lower_triangular(A, tol)
        real(dp), intent(in) :: A(:, :)
        real(dp), intent(in) :: tol
        integer :: n, i

        n = size(A, 1)

        if (size(A, 2) /= n) error stop "A must be square"
        if (tol < 0.0_dp) error stop "tol must be non-negative"

        is_lower_triangular = .true.

        do i = 2, n
        ! 第 i 欄、對角線右側：上三角元素
            if (any(abs(A(1:i-1, i)) > tol)) then
                is_lower_triangular = .false.
                return
            end if
        end do

    end function is_lower_triangular

    logical function dummy_select(wr, wi)
        real(dp), intent(in) :: wr, wi

        dummy_select = .false.

    end function dummy_select

    subroutine solve_lyapunov_triangular(A, C, X, max_real_part)
    ! Solve A X + X A^T = C, where A is upper or lower triangular.
        integer :: n, i, info
        external :: dtrsyl
        real(dp), intent(in) :: A(:, :), C(:, :)
        real(dp), intent(out) :: X(:, :), max_real_part
        logical :: is_upper, is_lower
        real(dp) :: scale, tol
        real(dp), allocatable :: T(:, :)

        ! 1. Check dimensions
        n = size(A, 1)
        if (size(A, 2) /= n) error stop "A must be square"
        if (size(C, 1) /= n .or. size(C, 2) /= n) then
            error stop "C has wrong shape"
        end if
        if (size(X, 1) /= n .or. size(X, 2) /= n) then
            error stop "X has wrong shape"
        end if

        ! 2. Identify the triangular structure.
        tol = 100.0_dp * epsilon(1.0_dp) * max(1.0_dp, maxval(abs(A)))
        is_upper = is_upper_triangular(A, tol)
        is_lower = is_lower_triangular(A, tol)

        if (.not. is_upper .and. .not. is_lower) then
            error stop "A is not triangular"
        end if

        ! 3. For triangular A, eigenvalues are its diagonal entries.
        max_real_part = A(1, 1)
        do i = 2, n
            max_real_part = max(max_real_part, A(i, i))
        end do

        if (max_real_part >= 0.0_dp) then
            print *, "max Re(lambda(A)) =", max_real_part
            error stop "A is not Hurwitz stable"
        end if

        ! 4. DTRSYL overwrites its RHS, so X is its work array.
        X = C

        if (is_upper) then
            ! A X + X A^T = scale * C
            call dtrsyl('N', 'T', 1, n, n, A, n, A, n, X, n, scale, info)
        else
            ! T = A^T is upper triangular:
            ! T^T X + X T = scale * C
            allocate(T(n, n))
            T = transpose(A)
            call dtrsyl('T', 'N', 1, n, n, T, n, T, n, X, n, scale, info)
        end if

        if (info < 0) error stop "DTRSYL: illegal argument"
        if (info == 1) print *, "Warning: Lyapunov equation is nearly singular."

        if (scale <= tiny(1.0_dp)) then
            error stop "DTRSYL returned an invalid scale"
        end if
        X = X / scale

    end subroutine solve_lyapunov_triangular

    subroutine solve_lyapunov_triangular_blocked(A, C, X, max_real_part, &
                                                 is_upper, is_lower)
    ! Solve A X + X A^T = C with the blocked LAPACK DTRSYL3 routine.
        integer :: n, i, info
        integer :: liwork, ldswork, nswork
        integer :: iwork_query(1)
        integer, allocatable :: iwork(:)
        external :: dtrsyl3
        real(dp), intent(in) :: A(:, :), C(:, :)
        real(dp), intent(out) :: X(:, :), max_real_part
        logical, intent(in) :: is_upper, is_lower
        real(dp) :: scale
        real(dp) :: swork_query(2, 1)
        real(dp), allocatable :: T(:, :), swork(:, :)

        n = size(A, 1)
        if (size(A, 2) /= n) error stop "A must be square"
        if (size(C, 1) /= n .or. size(C, 2) /= n) then
            error stop "C has wrong shape"
        end if
        if (size(X, 1) /= n .or. size(X, 2) /= n) then
            error stop "X has wrong shape"
        end if

        if (.not. is_upper .and. .not. is_lower) then
            error stop "Triangular solver requires upper or lower A"
        end if

        max_real_part = A(1, 1)
        do i = 2, n
            max_real_part = max(max_real_part, A(i, i))
        end do

        if (max_real_part >= 0.0_dp) then
            print *, "max Re(lambda(A)) =", max_real_part
            error stop "A is not Hurwitz stable"
        end if

        if (is_lower .and. .not. is_upper) then
            allocate(T(n, n))
            T = transpose(A)
        end if

        ! Query the optimal integer and floating-point workspaces.
        liwork = -1
        ldswork = -1
        X = C

        if (is_upper) then
            call dtrsyl3('N', 'T', 1, n, n, A, n, A, n, X, n, &
                         scale, iwork_query, liwork, swork_query, &
                         ldswork, info)
        else
            call dtrsyl3('T', 'N', 1, n, n, T, n, T, n, X, n, &
                         scale, iwork_query, liwork, swork_query, &
                         ldswork, info)
        end if

        if (info /= 0) error stop "DTRSYL3 workspace query failed"

        liwork = max(1, iwork_query(1))
        ldswork = max(2, int(swork_query(1, 1)))
        nswork = max(1, int(swork_query(2, 1)))
        allocate(iwork(liwork), swork(ldswork, nswork))

        ! DTRSYL3 overwrites the right-hand side with the solution.
        X = C

        if (is_upper) then
            call dtrsyl3('N', 'T', 1, n, n, A, n, A, n, X, n, &
                         scale, iwork, liwork, swork, ldswork, info)
        else
            call dtrsyl3('T', 'N', 1, n, n, T, n, T, n, X, n, &
                         scale, iwork, liwork, swork, ldswork, info)
        end if

        if (info < 0) error stop "DTRSYL3: illegal argument"
        if (info == 1) then
            print *, "Warning: blocked Lyapunov equation is nearly singular."
        end if

        if (scale <= tiny(1.0_dp)) then
            error stop "DTRSYL3 returned an invalid scale"
        end if
        X = X / scale

    end subroutine solve_lyapunov_triangular_blocked

    subroutine solve_lyapunov(A, C, X, max_real_part)
    ! solve AX + XA^T = C
        integer :: n, lwork
        integer :: info, sdim
        external :: dgemm
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
            print *, "max Re(lambda(A)) =", max_real_part
            error stop "A is not Hurwitz stable"
        end if

        ! -------------------------------------------------
        ! 3. F = U^T C U
        ! -------------------------------------------------

        call dgemm('T', 'N', n, n, n, 1.0_dp, U, n, C, n, &
                   0.0_dp, tmp, n)
        call dgemm('N', 'N', n, n, n, 1.0_dp, tmp, n, U, n, &
                   0.0_dp, F, n)

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

        call dgemm('N', 'N', n, n, n, 1.0_dp, U, n, Y, n, &
                   0.0_dp, tmp, n)
        call dgemm('N', 'T', n, n, n, 1.0_dp, tmp, n, U, n, &
                   0.0_dp, X, n)

    end subroutine solve_lyapunov

    subroutine solve_lyapunov_blocked(A, C, X, max_real_part)
    ! Solve A X + X A^T = C using real Schur form and DTRSYL3.
        integer :: n, lwork
        integer :: info, sdim
        integer :: liwork, ldswork, nswork
        integer :: iwork_query(1)
        external :: dgemm, dtrsyl3
        real(dp), intent(in) :: A(:, :)
        real(dp), intent(in) :: C(:, :)
        real(dp), intent(out) :: X(:, :), max_real_part

        real(dp) :: scale
        real(dp) :: swork_query(2, 1)

        real(dp), allocatable :: T(:, :)
        real(dp), allocatable :: U(:, :)
        real(dp), allocatable :: Y(:, :)
        real(dp), allocatable :: F(:, :)
        real(dp), allocatable :: tmp(:, :)
        real(dp), allocatable :: wr(:), wi(:)
        real(dp), allocatable :: work(:)
        real(dp), allocatable :: swork(:, :)
        integer, allocatable :: iwork(:)
        logical, allocatable :: bwork(:)

        n = size(A, 1)

        if (size(A, 2) /= n) error stop "A must be square"
        if (size(C, 1) /= n .or. size(C, 2) /= n) then
            error stop "C has wrong shape"
        end if
        if (size(X, 1) /= n .or. size(X, 2) /= n) then
            error stop "X has wrong shape"
        end if

        allocate(T(n, n), U(n, n), F(n, n), Y(n, n), tmp(n, n))
        allocate(wr(n), wi(n), bwork(n))

        ! Workspace query for the real Schur decomposition.
        T = A
        lwork = -1
        allocate(work(1))

        call dgees('V', 'N', dummy_select, n, T, n, &
                   sdim, wr, wi, U, n, work, lwork, bwork, info)

        if (info /= 0) error stop "DGEES workspace query failed"

        lwork = int(work(1))
        deallocate(work)
        allocate(work(lwork))

        ! Real Schur decomposition: A = U T U^T.
        T = A
        call dgees('V', 'N', dummy_select, n, T, n, &
                   sdim, wr, wi, U, n, work, lwork, bwork, info)

        if (info /= 0) error stop "DGEES failed"

        max_real_part = maxval(wr)
        if (max_real_part >= 0.0_dp) then
            print *, "max Re(lambda(A)) =", max_real_part
            error stop "A is not Hurwitz stable"
        end if

        ! Transform the right-hand side: F = U^T C U.
        call dgemm('T', 'N', n, n, n, 1.0_dp, U, n, C, n, &
                   0.0_dp, tmp, n)
        call dgemm('N', 'N', n, n, n, 1.0_dp, tmp, n, U, n, &
                   0.0_dp, F, n)

        ! Query the optimal DTRSYL3 workspaces.
        liwork = -1
        ldswork = -1
        Y = F

        call dtrsyl3('N', 'T', 1, n, n, T, n, T, n, Y, n, &
                     scale, iwork_query, liwork, swork_query, &
                     ldswork, info)

        if (info /= 0) error stop "DTRSYL3 workspace query failed"

        liwork = max(1, iwork_query(1))
        ldswork = max(2, int(swork_query(1, 1)))
        nswork = max(1, int(swork_query(2, 1)))
        allocate(iwork(liwork), swork(ldswork, nswork))

        ! Solve T Y + Y T^T = scale * F.
        Y = F
        call dtrsyl3('N', 'T', 1, n, n, T, n, T, n, Y, n, &
                     scale, iwork, liwork, swork, ldswork, info)

        if (info < 0) error stop "DTRSYL3: illegal argument"
        if (info == 1) then
            print *, "Warning: blocked Lyapunov equation is nearly singular."
        end if

        if (scale <= tiny(1.0_dp)) then
            error stop "DTRSYL3 returned an invalid scale"
        end if
        Y = Y / scale

        ! Transform back: X = U Y U^T.
        call dgemm('N', 'N', n, n, n, 1.0_dp, U, n, Y, n, &
                   0.0_dp, tmp, n)
        call dgemm('N', 'T', n, n, n, 1.0_dp, tmp, n, U, n, &
                   0.0_dp, X, n)

    end subroutine solve_lyapunov_blocked

    subroutine compute_lyapunov_residual(Q, K, noise, q_is_upper, &
                                         q_is_lower, max_residual)
        ! Compute max|Q K + K Q^T + noise|.
        integer :: n
        real(dp), intent(in) :: Q(:, :), K(:, :), noise(:, :)
        logical, intent(in) :: q_is_upper, q_is_lower
        real(dp), intent(out) :: max_residual
        real(dp), allocatable :: residual(:, :), second_term(:, :)
        external :: dtrmm, dgemm

        n = size(Q, 1)

        if (size(Q, 2) /= n) error stop "Q must be square"
        if (size(K, 1) /= n .or. size(K, 2) /= n) then
            error stop "Q and K size mismatch"
        end if
        if (size(noise, 1) /= n .or. size(noise, 2) /= n) then
            error stop "Q and noise size mismatch"
        end if

        allocate(residual(n, n))

        if (q_is_upper) then
            ! residual = Q K
            residual = K
            call dtrmm('L', 'U', 'N', 'N', n, n, 1.0_dp, &
                       Q, n, residual, n)

            ! second_term = K Q^T
            allocate(second_term(n, n))
            second_term = K
            call dtrmm('R', 'U', 'T', 'N', n, n, 1.0_dp, &
                       Q, n, second_term, n)

            residual = residual + second_term

        else if (q_is_lower) then
            ! residual = Q K
            residual = K
            call dtrmm('L', 'L', 'N', 'N', n, n, 1.0_dp, &
                       Q, n, residual, n)

            ! second_term = K Q^T
            allocate(second_term(n, n))
            second_term = K
            call dtrmm('R', 'L', 'T', 'N', n, n, 1.0_dp, &
                       Q, n, second_term, n)

            residual = residual + second_term

        else
            ! General dense Q: residual = Q K + K Q^T
            call dgemm('N', 'N', n, n, n, 1.0_dp, Q, n, K, n, &
                       0.0_dp, residual, n)
            call dgemm('N', 'T', n, n, n, 1.0_dp, K, n, Q, n, &
                       1.0_dp, residual, n)
        end if

        residual = residual + noise
        max_residual = maxval(abs(residual))

    end subroutine compute_lyapunov_residual

    subroutine analytic_result(Q, noise, K, alpha, q_is_upper, q_is_lower)
        ! Analytic result for alpha.
        integer :: n
        real(dp), intent(in) :: Q(:, :), noise(:, :), K(:, :)
        real(dp), allocatable, intent(out) :: alpha(:, :)
        logical, intent(in) :: q_is_upper, q_is_lower
        external :: dtrmm, dgemm

        n = size(Q, 1)

        if (size(Q, 2) /= n) error stop "Q must be square"
        if (size(K, 1) /= n .or. size(K, 2) /= n) then
            error stop "Q and K size mismatch"
        end if
        if (size(noise, 1) /= n .or. size(noise, 2) /= n) then
            error stop "Q and noise size mismatch"
        end if

        allocate(alpha(n, n))

        if (q_is_upper) then
        ! A = Q is upper triangular, so we can use DTRMM to compute alpha = -2 Q K
            alpha = K
            call dtrmm('L', 'U', 'N', 'N', n, n, -2.0_dp, &
                       Q, n, alpha, n)
        else if (q_is_lower) then
        ! A = Q is lower triangular, so we can use DTRMM to compute alpha = -2 Q K
            alpha = K
            call dtrmm('L', 'L', 'N', 'N', n, n, -2.0_dp, &
                       Q, n, alpha, n)
        else
        ! A = Q is not triangular, so we use DGEMM to compute alpha = -2 Q K
            call dgemm('N', 'N', n, n, n, -2.0_dp, Q, n, K, n, &
                       0.0_dp, alpha, n)
        end if

        alpha = alpha - noise

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

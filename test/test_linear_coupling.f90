program test_linear_coupling
    use precision_mod
    use langevin_mod, only : construct_Q, compute_force
    use theory_mod, only : solve_fixed_point_linear, &
        solve_lyapunov_triangular_blocked, analytic_result, compute_energetics_theory
    use shuffle_mod, only : run_shuffle_ensemble
    implicit none
    real(dp) :: r(2), W(2,2), bias(2), x(2), force(2), fixedpoint(2)
    real(dp) :: noise(2,2), K(2,2), expected_K(2,2), max_real_part
    real(dp), allocatable :: Q(:,:), alpha(:,:), heat(:), work(:), internal(:), entropy(:)
    logical :: adjacency(2,2)
    integer :: unit, status, trial, trial_id
    real(dp) :: trial_eigenvalue, trial_entropy
    character(len=512) :: line
    real(dp), parameter :: tol = 1.0e-12_dp

    ! dx1 = -2*x1 + 1 + noise; dx2 = -2*x2 + 3*x1 - 1 + noise.
    r = 2.0_dp
    W = 0.0_dp
    W(2,1) = 3.0_dp
    bias = [1.0_dp, -1.0_dp]
    x = [0.4_dp, -0.2_dp]
    call construct_Q(r, W, "LINEAR", Q)
    call compute_force(x, r, W, bias, "LINEAR", force)
    if (maxval(abs(force - [0.2_dp, 0.6_dp])) > tol) error stop "LINEAR force"
    if (maxval(abs(force - matmul(Q,x) - bias)) > tol) error stop "LINEAR Q"
    call solve_fixed_point_linear(Q, bias, .false., .true., fixedpoint)
    if (maxval(abs(fixedpoint - [0.5_dp, 0.25_dp])) > tol) error stop "LINEAR fixed point"

    noise = 0.0_dp
    noise(1,1) = 0.4_dp
    noise(2,2) = 0.4_dp
    ! For Q=[-a,0;c,-a], K11=sigma/(2a), K12=c*K11/(2a),
    ! K22=K11+c*K12/a, and total entropy=c**2/(2a).
    expected_K = reshape([0.1_dp, 0.075_dp, 0.075_dp, 0.2125_dp], [2,2])
    call solve_lyapunov_triangular_blocked(Q, -noise, K, max_real_part, .false., .true.)
    if (maxval(abs(K-expected_K)) > tol) error stop "LINEAR analytic covariance"
    call analytic_result(Q, noise, K, alpha, .false., .true.)
    call compute_energetics_theory(Q, noise, alpha, heat, work, internal, entropy)
    if (abs(sum(entropy)-2.25_dp) > tol) error stop "LINEAR analytic entropy"

    ! Bias permutation must not change LINEAR theory energetics.
    adjacency = .false.
    adjacency(2,1) = .true.
    call run_shuffle_ensemble(adjacency, W, bias, r, noise, "LINEAR", "BOTH", &
        .false., .true., 2, 2718, tol, 100, sum(entropy), -2.0_dp, &
        "test_linear_stability.csv", "test_linear_energetics.csv", "test_linear_summary.csv")
    open(newunit=unit, file="test_linear_energetics.csv", status="old")
    read(unit, '(A)') line
    do trial = 1, 2
        read(unit, *, iostat=status) trial_id, trial_eigenvalue, trial_entropy
        if (status /= 0) error stop "LINEAR shuffle output missing"
        if (trial_id /= trial .or. abs(trial_eigenvalue+2.0_dp) > tol) &
            error stop "LINEAR shuffle identifiers or stability"
        if (abs(trial_entropy-2.25_dp) > tol) error stop "LINEAR shuffle changed entropy"
    end do
    close(unit, status="delete")
    open(newunit=unit, file="test_linear_stability.csv", status="old")
    close(unit, status="delete")
    open(newunit=unit, file="test_linear_summary.csv", status="old")
    close(unit, status="delete")
    print *, "LINEAR coupling tests passed."
end program test_linear_coupling

module shuffle_mod
    use precision_mod
    use random_mod, only : initialize_seed
    use network_mod, only : shuffle_FCNN_weights
    use langevin_mod, only : construct_Q
    use theory_mod, only : &
        solve_lyapunov_triangular_blocked, &
        analytic_result, &
        compute_energetics_theory

    implicit none
    private

    public :: run_shuffle_ensemble

contains

    subroutine run_shuffle_ensemble( &
    adj_matrix, W_original, r, noise, &
    q_is_upper, q_is_lower, &
    n_shuffle, shuffle_seed, &
    original_total_entropy, &
    stability_filename, energetics_filename, &
    summary_filename)

    logical, intent(in) :: adj_matrix(:, :)
    real(dp), intent(in) :: W_original(:, :)
    real(dp), intent(in) :: r(:)
    real(dp), intent(in) :: noise(:, :)
    real(dp), intent(in) :: original_total_entropy

    logical, intent(in) :: q_is_upper
    logical, intent(in) :: q_is_lower

    integer, intent(in) :: n_shuffle
    integer, intent(in) :: shuffle_seed

    character(len=*), intent(in) :: stability_filename, energetics_filename, &
        summary_filename

    integer :: n, i
    integer :: shuffle_id
    integer :: stability_unit, energetics_unit, summary_unit
    integer :: io_status
    integer :: n_stable, n_marginal, n_unstable

    real(dp) :: stability_tol
    real(dp) :: max_real_part
    real(dp) :: stability_gap

    real(dp), allocatable :: W_trial(:, :)
    real(dp), allocatable :: Q_trial(:, :)
    real(dp), allocatable :: rhs(:, :)
    real(dp), allocatable :: K_trial(:, :)
    real(dp), allocatable :: alpha_trial(:, :)

    real(dp), allocatable :: heat_trial(:)
    real(dp), allocatable :: entropy_trial(:)
    real(dp), allocatable :: work_trial(:)
    real(dp), allocatable :: internal_trial(:)

    integer :: n_entropy_le_original

    real(dp) :: total_entropy_trial
    real(dp) :: entropy_mean
    real(dp) :: entropy_m2
    real(dp) :: entropy_delta
    real(dp) :: entropy_std
    real(dp) :: entropy_min
    real(dp) :: entropy_max
    real(dp) :: original_percentile
    real(dp) :: entropy_z_score

    n = size(W_original, 1)

    if (n <= 0) then
        error stop "Network size must be positive"
    end if

    if (size(W_original, 2) /= n) then
        error stop "W_original must be square"
    end if

    if (size(adj_matrix, 1) /= n .or. &
        size(adj_matrix, 2) /= n) then
        error stop "adj_matrix and W_original size mismatch"
    end if

    if (size(r) /= n) then
        error stop "r and W_original size mismatch"
    end if

    if (size(noise, 1) /= n .or. &
        size(noise, 2) /= n) then
        error stop "noise and W_original size mismatch"
    end if

    if (n_shuffle < 0) then
        error stop "n_shuffle must be non-negative"
    end if

    if (n_shuffle == 0) return

    if (shuffle_seed < 0) then
        error stop "shuffle_seed must be non-negative"
    end if

    if (.not. q_is_upper .and. .not. q_is_lower) then
        error stop "shuffle_mod currently requires triangular FCNN Q"
    end if

    if (len_trim(stability_filename) == 0 .or. &
        len_trim(energetics_filename) == 0 .or. &
        len_trim(summary_filename) == 0) then
        error stop "Shuffle output filenames must not be empty"
    end if

    ! Allocate reusable arrays
    allocate(W_trial(n, n))
    allocate(rhs(n, n))
    allocate(K_trial(n, n))

    ! Constant Lyapunov right-hand side
    rhs = -noise

    ! Independent RNG stream for weight shuffling
    call initialize_seed(shuffle_seed)

    open( &
    newunit=stability_unit, &
    file=trim(stability_filename), &
    status="replace", &
    action="write", &
    iostat=io_status)

    if (io_status /= 0) then
        error stop "Cannot open shuffle stability output"
    end if

    write(stability_unit, '(A)') &
    "shuffle_id,status,max_real_part,stability_gap"

    open( &
    newunit=energetics_unit, &
    file=trim(energetics_filename), &
    status="replace", &
    action="write", &
    iostat=io_status)

    if (io_status /= 0) then
        error stop "Cannot open shuffle energetics output"
    end if

    write(energetics_unit, '(A)') &
    "shuffle_id,max_real_part,total_entropy," // &
    "total_heat,total_work,total_internal"

    n_stable = 0
    n_marginal = 0
    n_unstable = 0

    entropy_mean = 0.0_dp
    entropy_m2 = 0.0_dp

    entropy_min = huge(1.0_dp)
    entropy_max = -huge(1.0_dp)

    n_entropy_le_original = 0

    do shuffle_id = 1, n_shuffle

    ! 每次都從相同的原始權重開始
    W_trial = W_original

    ! 只排列既有 edge 的權重
    call shuffle_FCNN_weights(adj_matrix, W_trial)

    ! 所有 shuffle 使用相同的 r
    call construct_Q(r, W_trial, "DIFFUSIVE", Q_trial)

    ! FCNN Q 是 triangular，所以 eigenvalues 是 diagonal
    max_real_part = Q_trial(1,1)

    do i = 2, n
        max_real_part = max( &
            max_real_part, Q_trial(i,i))
    end do

    stability_tol = 100.0_dp * epsilon(1.0_dp) * &
                    max(1.0_dp, maxval(abs(Q_trial)))

    stability_gap = -max_real_part

    if (max_real_part > stability_tol) then

        n_unstable = n_unstable + 1

        write(stability_unit, '(*(G0,:,","))') &
            shuffle_id, "unstable", &
            max_real_part, stability_gap

        cycle

    else if (max_real_part >= -stability_tol) then

        n_marginal = n_marginal + 1

        write(stability_unit, '(*(G0,:,","))') &
            shuffle_id, "marginal", &
            max_real_part, stability_gap

        cycle

    end if

    ! 只有到這裡的網路才是 stable
    n_stable = n_stable + 1

    write(stability_unit, '(*(G0,:,","))') &
        shuffle_id, "stable", &
        max_real_part, stability_gap

    ! Solve steady-state covariance
    call solve_lyapunov_triangular_blocked( &
    Q_trial, rhs, K_trial, max_real_part, &
    q_is_upper, q_is_lower)

    ! Compute analytic irreversibility matrix alpha
    call analytic_result( &
    Q_trial, noise, K_trial, alpha_trial, &
    q_is_upper, q_is_lower)

    ! Compute analytic energetics
    call compute_energetics_theory( &
    Q_trial, noise, alpha_trial, &
    heat_trial, work_trial, &
    internal_trial, entropy_trial)

    total_entropy_trial = sum(entropy_trial)

    ! Online ensemble statistics for stable networks (Welford algorithm)
    entropy_delta = total_entropy_trial - entropy_mean
    entropy_mean = entropy_mean + &
        entropy_delta / real(n_stable, dp)
    entropy_m2 = entropy_m2 + entropy_delta * &
        (total_entropy_trial - entropy_mean)

    entropy_min = min(entropy_min, total_entropy_trial)
    entropy_max = max(entropy_max, total_entropy_trial)

    if (total_entropy_trial <= original_total_entropy) then
        n_entropy_le_original = n_entropy_le_original + 1
    end if

    ! Write total energetics of this stable network
    write(energetics_unit, '(*(G0,:,","))') &
    shuffle_id, max_real_part, &
    total_entropy_trial, &
    sum(heat_trial), &
    sum(work_trial), &
    sum(internal_trial)

    end do

    close(stability_unit)
    close(energetics_unit)

    open( &
    newunit=summary_unit, &
    file=trim(summary_filename), &
    status="replace", &
    action="write", &
    iostat=io_status)

    if (io_status /= 0) then
        error stop "Cannot open shuffle summary output"
    end if

    write(summary_unit, '(A)') &
        "n_requested,n_stable,n_marginal,n_unstable," // &
        "original_entropy,mean_entropy,std_entropy," // &
        "min_entropy,max_entropy,percentile,z_score"

    if (n_stable >= 2) then
        entropy_std = sqrt(max(0.0_dp, entropy_m2) / &
            real(n_stable - 1, dp))
        original_percentile = 100.0_dp * &
            real(n_entropy_le_original, dp) / real(n_stable, dp)

        if (entropy_std > 0.0_dp) then
            entropy_z_score = &
                (original_total_entropy - entropy_mean) / entropy_std

            write(summary_unit, '(*(G0,:,","))') &
                n_shuffle, n_stable, n_marginal, n_unstable, &
                original_total_entropy, entropy_mean, entropy_std, &
                entropy_min, entropy_max, original_percentile, &
                entropy_z_score
        else
            write(summary_unit, '(*(G0,:,","))') &
                n_shuffle, n_stable, n_marginal, n_unstable, &
                original_total_entropy, entropy_mean, entropy_std, &
                entropy_min, entropy_max, original_percentile, "NA"
        end if

    else if (n_stable == 1) then
        original_percentile = 100.0_dp * &
            real(n_entropy_le_original, dp)

        write(summary_unit, '(*(G0,:,","))') &
            n_shuffle, n_stable, n_marginal, n_unstable, &
            original_total_entropy, entropy_mean, "NA", &
            entropy_min, entropy_max, original_percentile, "NA"

    else
        write(summary_unit, '(*(G0,:,","))') &
            n_shuffle, n_stable, n_marginal, n_unstable, &
            original_total_entropy, &
            "NA", "NA", "NA", "NA", "NA", "NA"
    end if

    close(summary_unit)

    print *, "-------------------------------"
    print *, "Weight-shuffle stability"
    print *, "-------------------------------"
    print *, "Total shuffles    =", n_shuffle
    print *, "Stable networks   =", n_stable
    print *, "Marginal networks =", n_marginal
    print *, "Unstable networks =", n_unstable

    if (n_stable > 0) then
        print *, "Original entropy  =", original_total_entropy
        print *, "Shuffle mean      =", entropy_mean
        print *, "Shuffle range     =", entropy_min, entropy_max
        print *, "Original percentile (%) =", original_percentile

        if (n_stable >= 2) then
            print *, "Shuffle std       =", entropy_std
            if (entropy_std > 0.0_dp) then
                print *, "Original z-score  =", entropy_z_score
            else
                print *, "Original z-score  = NA (zero shuffle variance)"
            end if
        else
            print *, "Shuffle std       = NA (fewer than two stable networks)"
            print *, "Original z-score  = NA"
        end if
    else
        print *, "Ensemble summary  = NA (no stable networks)"
    end if

    end subroutine run_shuffle_ensemble


end module shuffle_mod

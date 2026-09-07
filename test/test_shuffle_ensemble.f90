program test_shuffle_ensemble
    use precision_mod
    use random_mod, only : initialize_seed
    use shuffle_mod, only : run_shuffle_ensemble, &
        shuffle_bias_values
    implicit none

    integer, parameter :: n = 4
    character(len=*), parameter :: stability_file = &
        "test_shuffle_stability.csv"
    character(len=*), parameter :: energetics_file = &
        "test_shuffle_energetics.csv"
    character(len=*), parameter :: summary_file = &
        "test_shuffle_summary.csv"
    logical :: adjacency(n, n)
    logical :: file_exists
    integer :: i, io_unit, io_status, shuffle_id
    character(len=16) :: status_label
    character(len=256) :: header
    real(dp) :: W(n, n), bias(n), shuffled_bias(n)
    real(dp) :: r(n), noise(n, n)
    real(dp) :: max_real_part
    real(dp) :: min_kappa_in, min_kappa_out
    real(dp) :: original_entropy
    real(dp) :: original_min_kappa_in, original_min_kappa_out
    real(dp) :: original_max_real_part
    integer :: n_requested, n_stable, n_marginal, n_unstable

    bias = [1.0_dp, 2.0_dp, 10.0_dp, 20.0_dp]
    shuffled_bias = bias

    call initialize_seed(314)
    call shuffle_bias_values(shuffled_bias)

    if (abs(sum(shuffled_bias) - sum(bias)) > epsilon(1.0_dp) .or. &
        abs(sum(shuffled_bias**2) - sum(bias**2)) > &
        10.0_dp * epsilon(1.0_dp)) then
        error stop "Bias shuffle changed the network-wide bias multiset"
    end if

    adjacency = .false.
    W = 0.0_dp
    adjacency(3, 1) = .true.
    adjacency(3, 2) = .true.
    adjacency(4, 1) = .true.
    adjacency(4, 2) = .true.
    W(3, 1) = 0.10_dp
    W(3, 2) = -0.05_dp
    W(4, 1) = 0.08_dp
    W(4, 2) = 0.03_dp

    r = 2.0_dp
    bias = [0.0_dp, 0.0_dp, 0.1_dp, -0.1_dp]
    noise = 0.0_dp
    do i = 1, n
        noise(i, i) = 0.2_dp
    end do

    call run_shuffle_ensemble( &
        adjacency, W, bias, &
        r, noise, "TANH", "BOTH", &
        .false., .true., 2, 2718, &
        1.0e-12_dp, 100, 0.0_dp, -2.0_dp, &
        stability_file, energetics_file, summary_file)

    inquire(file=stability_file, exist=file_exists)
    if (.not. file_exists) error stop "Missing shuffle stability output"

    open(newunit=io_unit, file=stability_file, status="old", &
         action="read", iostat=io_status)
    if (io_status /= 0) error stop "Cannot read shuffle stability output"

    read(io_unit, '(A)', iostat=io_status) header
    if (io_status /= 0 .or. trim(header) /= &
        "shuffle_id,status,max_real_part,min_kappa_in,min_kappa_out") then
        error stop "Incorrect shuffle stability header"
    end if

    read(io_unit, *, iostat=io_status) &
        shuffle_id, status_label, max_real_part, &
        min_kappa_in, min_kappa_out
    if (io_status /= 0) error stop "Cannot parse shuffle stability row"
    if (shuffle_id /= 1 .or. trim(status_label) /= "stable") then
        error stop "Incorrect shuffle stability identifiers"
    end if
    if (min_kappa_in < -0.02_dp - 10.0_dp * epsilon(1.0_dp) .or. &
        min_kappa_in > 0.05_dp + 10.0_dp * epsilon(1.0_dp) .or. &
        min_kappa_out < -0.02_dp - 10.0_dp * epsilon(1.0_dp) .or. &
        min_kappa_out > 0.05_dp + 10.0_dp * epsilon(1.0_dp)) then
        error stop "Invalid shuffled in/out-strength minima"
    end if
    close(io_unit)

    inquire(file=energetics_file, exist=file_exists)
    if (.not. file_exists) error stop "Missing shuffle energetics output"
    inquire(file=summary_file, exist=file_exists)
    if (.not. file_exists) error stop "Missing shuffle summary output"

    open(newunit=io_unit, file=summary_file, status="old", &
         action="read", iostat=io_status)
    if (io_status /= 0) error stop "Cannot read shuffle summary output"

    read(io_unit, '(A)', iostat=io_status) header
    if (io_status /= 0 .or. trim(header) /= &
        "n_requested,n_stable,n_marginal,n_unstable," // &
        "original_entropy,original_min_kappa_in," // &
        "original_min_kappa_out,original_max_real_part") then
        error stop "Incorrect shuffle summary header"
    end if

    read(io_unit, *, iostat=io_status) &
        n_requested, n_stable, n_marginal, n_unstable, &
        original_entropy, original_min_kappa_in, &
        original_min_kappa_out, original_max_real_part
    if (io_status /= 0) error stop "Cannot parse shuffle summary row"
    if (n_requested /= 2 .or. n_stable /= 2 .or. &
        n_marginal /= 0 .or. n_unstable /= 0) then
        error stop "Incorrect shuffle summary counts"
    end if
    if (abs(original_entropy) > epsilon(1.0_dp) .or. &
        abs(original_min_kappa_in) > epsilon(1.0_dp) .or. &
        abs(original_min_kappa_out + 0.02_dp) > &
        10.0_dp * epsilon(1.0_dp) .or. &
        abs(original_max_real_part + 2.0_dp) > &
        10.0_dp * epsilon(1.0_dp)) then
        error stop "Incorrect original network summary values"
    end if
    close(io_unit)

    call delete_test_file(stability_file)
    call delete_test_file(energetics_file)
    call delete_test_file(summary_file)

    print *, "TANH weight/bias shuffle ensemble test passed."

contains

    subroutine delete_test_file(filename)
        character(len=*), intent(in) :: filename
        integer :: io_unit, io_status

        open(newunit=io_unit, file=filename, status="old", &
             action="read", iostat=io_status)
        if (io_status /= 0) error stop "Cannot reopen shuffle test output"
        close(io_unit, status="delete")
    end subroutine delete_test_file

end program test_shuffle_ensemble

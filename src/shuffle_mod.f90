module shuffle_mod
    use precision_mod
    use random_mod, only : initialize_seed, rand_uniform
    use network_mod, only : &
    shuffle_FCNN_weights, &
    shuffle_fcnn_weights_by_layer
    use langevin_mod, only : construct_Q
    use theory_mod, only : &
        solve_lyapunov_triangular_blocked, &
        solve_fixed_point_nonlinear, &
        analytic_result, &
        compute_energetics_theory

    implicit none
    private

    public :: run_shuffle_ensemble
    public :: shuffle_bias_values

contains

    subroutine run_shuffle_ensemble( &
    adj_matrix, W_original, bias_original, &
    r, noise, coupling_type, shuffle_mode, &
    shuffle_scope, network_file, bias_file, &
    node_layer, q_is_upper, q_is_lower, &
    n_shuffle, shuffle_seed, &
    fixedpoint_tolerance, fixedpoint_max_iterations, &
    original_total_entropy, original_max_real_part, &
    stability_filename, energetics_filename, &
    summary_filename)

    logical, intent(in) :: adj_matrix(:, :)
    integer, intent(in) :: node_layer(:)
    real(dp), intent(in) :: W_original(:, :)
    real(dp), intent(in) :: bias_original(:)
    real(dp), intent(in) :: r(:)
    real(dp), intent(in) :: noise(:, :)
    character(len=*), intent(in) :: coupling_type, shuffle_mode, shuffle_scope
    character(len=*), intent(in) :: network_file, bias_file
    real(dp), intent(in) :: original_total_entropy
    real(dp), intent(in) :: original_max_real_part

    logical, intent(in) :: q_is_upper
    logical, intent(in) :: q_is_lower

    integer, intent(in) :: n_shuffle
    integer, intent(in) :: shuffle_seed
    real(dp), intent(in) :: fixedpoint_tolerance
    integer, intent(in) :: fixedpoint_max_iterations

    character(len=*), intent(in) :: stability_filename, energetics_filename, &
        summary_filename

    integer :: n, i
    integer :: shuffle_id
    integer :: stability_unit, energetics_unit, summary_unit
    integer :: io_status
    integer :: n_stable, n_marginal, n_unstable

    real(dp) :: stability_tol
    real(dp) :: max_real_part
    real(dp) :: min_kappa_in
    real(dp) :: min_kappa_out
    real(dp) :: original_min_kappa_in
    real(dp) :: original_min_kappa_out

    real(dp), allocatable :: W_trial(:, :)
    real(dp), allocatable :: bias_trial(:)
    real(dp), allocatable :: fixpoint_trial(:)
    real(dp), allocatable :: kappa_in(:)
    real(dp), allocatable :: kappa_out(:)
    real(dp), allocatable :: Q_trial(:, :)
    real(dp), allocatable :: rhs(:, :)
    real(dp), allocatable :: K_trial(:, :)
    real(dp), allocatable :: alpha_trial(:, :)

    real(dp), allocatable :: heat_trial(:)
    real(dp), allocatable :: entropy_trial(:)
    real(dp), allocatable :: work_trial(:)
    real(dp), allocatable :: internal_trial(:)

    logical :: check_weight_distribution
    logical :: check_bias_distribution

    real(dp) :: total_entropy_trial

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

    if (size(bias_original) /= n) then
        error stop "bias and W_original size mismatch"
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

    if (size(node_layer) /= n) then
        error stop "node_layer and network size mismatch"
    end if

    if (any(node_layer <= 0)) then
        error stop "Layer shuffle requires positive layer indices"
    end if

    select case (trim(adjustl(shuffle_scope)))
    case ("GLOBAL", "LAYER")
        continue
    case default
        error stop "shuffle_scope must be GLOBAL or LAYER"
    end select

    select case (trim(adjustl(shuffle_mode)))

    case ("WEIGHT")
    check_weight_distribution = .true.
    check_bias_distribution = .false.

    case ("BIAS")
    check_weight_distribution = .false.
    check_bias_distribution = .true.

    case ("BOTH")
    check_weight_distribution = .true.
    check_bias_distribution = .true.

    case default
    error stop "shuffle_mode must be WEIGHT, BIAS, or BOTH"

    end select

    select case (trim(adjustl(coupling_type)))
    case ("DIFFUSIVE", "LINEAR", "TANH", "TANH_INPUT")
        continue
    case default
        error stop "Unsupported coupling type in shuffle ensemble"
    end select

    if (fixedpoint_tolerance <= 0.0_dp) then
        error stop "Fixed-point tolerance must be positive"
    end if

    if (fixedpoint_max_iterations <= 0) then
        error stop "Fixed-point maximum iterations must be positive"
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
    allocate(bias_trial(n), fixpoint_trial(n))
    allocate(kappa_in(n), kappa_out(n))
    allocate(rhs(n, n))
    allocate(K_trial(n, n))

    ! Strength extrema of the unshuffled reference network
    kappa_in = 0.0_dp
    kappa_out = 0.0_dp
    do i = 1, n
        kappa_in = kappa_in + W_original(:, i)
        kappa_out = kappa_out + W_original(i, :)
    end do
    original_min_kappa_in = minval(kappa_in)
    original_min_kappa_out = minval(kappa_out)

    ! Constant Lyapunov right-hand side
    rhs = -noise

    ! Independent RNG stream for the selected shuffle operation
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
    "shuffle_id,status,max_real_part,min_kappa_in,min_kappa_out"

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

    do shuffle_id = 1, n_shuffle

    ! 每次都從相同的原始權重與 bias 開始
    W_trial = W_original
    bias_trial = bias_original

    select case (trim(adjustl(shuffle_scope)))

    case ("GLOBAL")

    select case (trim(adjustl(shuffle_mode)))
    case ("WEIGHT")
        call shuffle_fcnn_weights(adj_matrix, W_trial)

    case ("BIAS")
        call shuffle_bias_values(bias_trial)

    case ("BOTH")
        call shuffle_fcnn_weights(adj_matrix, W_trial)
        call shuffle_bias_values(bias_trial)
    end select

    case ("LAYER")

    select case (trim(adjustl(shuffle_mode)))
    case ("WEIGHT")
        call shuffle_fcnn_weights_by_layer( &
            adj_matrix, W_trial, node_layer)

    case ("BIAS")
        call shuffle_bias_values_by_layer( &
            bias_trial, node_layer)

    case ("BOTH")
        call shuffle_fcnn_weights_by_layer( &
            adj_matrix, W_trial, node_layer)

        call shuffle_bias_values_by_layer( &
            bias_trial, node_layer)
    end select

    end select

    ! Safety checks: a shuffle may change positions, but it must not
    ! change the topology or the network-wide value distributions.
    if (check_weight_distribution) then
        if (.not. weight_shuffle_is_valid( &
            adj_matrix, W_original, W_trial)) then
            error stop "Weight shuffle changed topology or distribution"
        end if
    end if

    if (check_bias_distribution) then
        if (.not. value_distribution_is_preserved( &
            bias_original, bias_trial)) then
            error stop "Bias shuffle changed its distribution"
        end if
    end if

    ! A global check is not sufficient for a layer-preserving shuffle:
    ! values could move between layers while keeping the network-wide
    ! distribution unchanged. Verify every affected layer separately.
    if (trim(adjustl(shuffle_scope)) == "LAYER") then
        if (check_weight_distribution) then
            call check_weight_statistics_by_layer( &
                adj_matrix, W_original, W_trial, node_layer, &
                shuffle_id == 1)
        end if

        if (check_bias_distribution) then
            call check_bias_statistics_by_layer( &
                bias_original, bias_trial, node_layer, &
                shuffle_id == 1)
        end if
    end if

    ! Signed weighted in-strength: W(i,j) represents j -> i.
    kappa_in = 0.0_dp
    kappa_out = 0.0_dp
    do i = 1, n
        kappa_in = kappa_in + W_trial(:, i)
        kappa_out = kappa_out + W_trial(i, :)
    end do
    min_kappa_in = minval(kappa_in)
    min_kappa_out = minval(kappa_out)

    ! 所有 shuffle 使用相同的 r 與 noise。TANH 必須先用 trial
    ! weights/bias 解新固定點，再於該固定點建立 Jacobian。
    select case (trim(adjustl(coupling_type)))
    case ("DIFFUSIVE", "LINEAR")
        call construct_Q(r, W_trial, coupling_type, Q_trial)
    case ("TANH", "TANH_INPUT")
        call solve_fixed_point_nonlinear( &
            r, W_trial, bias_trial, coupling_type, fixpoint_trial, &
            fixedpoint_tolerance, fixedpoint_max_iterations)
        call construct_Q( &
            r, W_trial, coupling_type, Q_trial, fixpoint_trial, bias_trial)
    end select

    ! FCNN Q 是 triangular，所以 eigenvalues 是 diagonal
    max_real_part = Q_trial(1,1)

    do i = 2, n
        max_real_part = max( &
            max_real_part, Q_trial(i,i))
    end do

    stability_tol = 100.0_dp * epsilon(1.0_dp) * &
                    max(1.0_dp, maxval(abs(Q_trial)))

    if (max_real_part > stability_tol) then

        n_unstable = n_unstable + 1

        write(stability_unit, '(*(G0,:,","))') &
            shuffle_id, "unstable", &
            max_real_part, min_kappa_in, min_kappa_out

        cycle

    else if (max_real_part >= -stability_tol) then

        n_marginal = n_marginal + 1

        write(stability_unit, '(*(G0,:,","))') &
            shuffle_id, "marginal", &
            max_real_part, min_kappa_in, min_kappa_out

        cycle

    end if

    ! 只有到這裡的網路才是 stable
    n_stable = n_stable + 1

    write(stability_unit, '(*(G0,:,","))') &
        shuffle_id, "stable", &
        max_real_part, min_kappa_in, min_kappa_out

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
        "coupling_type,shuffle_mode,network_file,bias_file," // &
        "n_requested,n_stable,n_marginal,n_unstable," // &
        "original_entropy,original_min_kappa_in," // &
        "original_min_kappa_out,original_max_real_part"

    write(summary_unit, '(A,",",A,",",A,",",A,",",*(G0,:,","))') &
        trim(adjustl(coupling_type)), trim(adjustl(shuffle_mode)), &
        trim(network_file), trim(bias_file), &
        n_shuffle, n_stable, n_marginal, n_unstable, &
        original_total_entropy, original_min_kappa_in, &
        original_min_kappa_out, original_max_real_part

    close(summary_unit)

    print *, "-------------------------------"
    print *, "Shuffle ensemble stability"
    print *, "-------------------------------"
    print *, "Total shuffles    =", n_shuffle
    print *, "Stable networks   =", n_stable
    print *, "Marginal networks =", n_marginal
    print *, "Unstable networks =", n_unstable
    print *, "Shuffle safety checks = passed"
    print *, "Original entropy  =", original_total_entropy
    print *, "Original min kappa_in  =", original_min_kappa_in
    print *, "Original min kappa_out =", original_min_kappa_out
    print *, "Original max Re(lambda) =", original_max_real_part

    end subroutine run_shuffle_ensemble

    subroutine calculate_mean_std(values, mean_value, std_value)
    real(dp), intent(in) :: values(:)
    real(dp), intent(out) :: mean_value
    real(dp), intent(out) :: std_value

    integer :: n
    real(dp) :: variance

    n = size(values)

    if (n <= 0) then
        error stop "Cannot compute statistics of an empty array"
    end if

    mean_value = sum(values) / real(n, dp)

    if (n == 1) then
        std_value = 0.0_dp
    else
        variance = sum((values - mean_value)**2) / &
                   real(n - 1, dp)

        ! 避免浮點誤差產生非常小的負數。
        std_value = sqrt(max(0.0_dp, variance))
    end if

    end subroutine calculate_mean_std

    subroutine check_mean_std(reference, trial, label, show_statistics)
    real(dp), intent(in) :: reference(:)
    real(dp), intent(in) :: trial(:)
    character(len=*), intent(in) :: label
    logical, intent(in) :: show_statistics

    real(dp) :: mean_reference, mean_trial
    real(dp) :: std_reference, std_trial
    real(dp) :: tolerance

    if (size(reference) /= size(trial)) then
        error stop trim(label) // ": array sizes differ"
    end if

    call calculate_mean_std( &
        reference, mean_reference, std_reference)

    call calculate_mean_std( &
        trial, mean_trial, std_trial)

    if (show_statistics) then
        print *, trim(label)
        print *, "  original mean =", mean_reference
        print *, "  shuffled mean =", mean_trial
        print *, "  original std  =", std_reference
        print *, "  shuffled std  =", std_trial
    end if

    tolerance = 100.0_dp * real(size(reference), dp) * &
                epsilon(1.0_dp) * &
                max(1.0_dp, abs(mean_reference), std_reference)

    if (abs(mean_trial - mean_reference) > tolerance) then
        print *, trim(label), " original mean =", mean_reference
        print *, trim(label), " shuffled mean =", mean_trial
        error stop "Shuffle changed the mean"
    end if

    if (abs(std_trial - std_reference) > tolerance) then
        print *, trim(label), " original std =", std_reference
        print *, trim(label), " shuffled std =", std_trial
        error stop "Shuffle changed the standard deviation"
    end if

    end subroutine check_mean_std

    subroutine check_weight_statistics_by_layer( &
        adj_matrix, W_reference, W_trial, node_layer, show_statistics)
        ! Check the mean and sample standard deviation independently for
        ! every source-layer -> target-layer connection that contains edges.
        logical, intent(in) :: adj_matrix(:, :)
        real(dp), intent(in) :: W_reference(:, :)
        real(dp), intent(in) :: W_trial(:, :)
        integer, intent(in) :: node_layer(:)
        logical, intent(in) :: show_statistics

        integer :: n, n_layers
        integer :: source_layer, target_layer
        integer :: source_node, target_node
        integer :: n_edges, edge_index
        real(dp), allocatable :: reference_values(:)
        real(dp), allocatable :: trial_values(:)
        character(len=64) :: label

        n = size(W_reference, 1)

        if (n <= 0 .or. size(W_reference, 2) /= n) then
            error stop "W_reference must be a nonempty square matrix"
        end if
        if (size(W_trial, 1) /= n .or. size(W_trial, 2) /= n) then
            error stop "W_reference and W_trial size mismatch"
        end if
        if (size(adj_matrix, 1) /= n .or. &
            size(adj_matrix, 2) /= n) then
            error stop "adj_matrix and W_reference size mismatch"
        end if
        if (size(node_layer) /= n .or. any(node_layer <= 0)) then
            error stop "Invalid node layers in weight statistics check"
        end if

        n_layers = maxval(node_layer)

        do source_layer = 1, n_layers - 1
            target_layer = source_layer + 1
            n_edges = 0

            do source_node = 1, n
                if (node_layer(source_node) /= source_layer) cycle

                do target_node = 1, n
                    if (node_layer(target_node) /= target_layer) cycle
                    if (adj_matrix(target_node, source_node)) &
                        n_edges = n_edges + 1
                end do
            end do

            if (n_edges == 0) cycle

            allocate(reference_values(n_edges), trial_values(n_edges))
            edge_index = 0

            do source_node = 1, n
                if (node_layer(source_node) /= source_layer) cycle

                do target_node = 1, n
                    if (node_layer(target_node) /= target_layer) cycle
                    if (.not. adj_matrix(target_node, source_node)) cycle

                    edge_index = edge_index + 1
                    reference_values(edge_index) = &
                        W_reference(target_node, source_node)
                    trial_values(edge_index) = &
                        W_trial(target_node, source_node)
                end do
            end do

            write(label, '(A,I0,A,I0)') &
                "Weight layers ", source_layer, " -> ", target_layer
            call check_mean_std( &
                reference_values, trial_values, label, show_statistics)

            deallocate(reference_values, trial_values)
        end do

    end subroutine check_weight_statistics_by_layer

    subroutine check_bias_statistics_by_layer( &
        bias_reference, bias_trial, node_layer, show_statistics)
        ! Check the mean and sample standard deviation independently for
        ! the bias values belonging to each node layer.
        real(dp), intent(in) :: bias_reference(:)
        real(dp), intent(in) :: bias_trial(:)
        integer, intent(in) :: node_layer(:)
        logical, intent(in) :: show_statistics

        integer :: n, n_layers, layer
        real(dp), allocatable :: reference_values(:)
        real(dp), allocatable :: trial_values(:)
        character(len=64) :: label

        n = size(bias_reference)

        if (n <= 0 .or. size(bias_trial) /= n) then
            error stop "bias_reference and bias_trial size mismatch"
        end if
        if (size(node_layer) /= n .or. any(node_layer <= 0)) then
            error stop "Invalid node layers in bias statistics check"
        end if

        n_layers = maxval(node_layer)

        do layer = 1, n_layers
            reference_values = pack( &
                bias_reference, node_layer == layer)
            trial_values = pack( &
                bias_trial, node_layer == layer)

            if (size(reference_values) == 0) then
                error stop "A node layer contains no bias values"
            end if

            write(label, '(A,I0)') "Bias layer ", layer
            call check_mean_std( &
                reference_values, trial_values, label, show_statistics)

            deallocate(reference_values, trial_values)
        end do

    end subroutine check_bias_statistics_by_layer

    subroutine shuffle_bias_values(bias)
        ! Preserve the network-wide bias multiset while allowing values
        ! to move between any nodes, including nodes in different layers.
        real(dp), intent(inout) :: bias(:)
        integer :: n, k, random_index
        real(dp) :: temp

        n = size(bias)

        if (n <= 0) error stop "Bias shuffle requires at least one node"

        ! One network-wide Fisher-Yates shuffle.
        do k = n, 2, -1
            random_index = 1 + &
                int(rand_uniform() * real(k, dp))

            temp = bias(k)
            bias(k) = bias(random_index)
            bias(random_index) = temp
        end do

    end subroutine shuffle_bias_values

    subroutine shuffle_bias_values_by_layer(bias, node_layer)

    real(dp), intent(inout) :: bias(:)
    integer, intent(in) :: node_layer(:)
    integer :: n, n_layers
    integer :: layer, i, k
    integer :: n_nodes_in_layer
    integer :: random_index

    integer, allocatable :: layer_nodes(:)
    real(dp) :: temp

    n = size(bias)
    if (size(bias) <= 0) then
    error stop "Bias array must not be empty"
    end if

    if (size(node_layer) /= size(bias)) then
    error stop "bias and node_layer size mismatch"
    end if

    if (any(node_layer <= 0)) then
    error stop "Layer indices must be positive"
    end if

    n_layers = maxval(node_layer)

    do layer = 1, n_layers

        n_nodes_in_layer = count(node_layer == layer)

        if (n_nodes_in_layer <= 1) cycle

        allocate(layer_nodes(n_nodes_in_layer))

        layer_nodes = pack( &
        [(i, i = 1, n)], &
        node_layer == layer)

        ! 只交換這個 layer 中的 bias。
        do k = n_nodes_in_layer, 2, -1
            random_index = 1 + &
            int(rand_uniform() * real(k, dp))

            temp = bias(layer_nodes(k))
            bias(layer_nodes(k)) = &
            bias(layer_nodes(random_index))
            bias(layer_nodes(random_index)) = temp
        end do

        deallocate(layer_nodes)

    end do

    end subroutine shuffle_bias_values_by_layer

    logical function weight_shuffle_is_valid( &
        adj_matrix, W_reference, W_trial)
        logical, intent(in) :: adj_matrix(:, :)
        real(dp), intent(in) :: W_reference(:, :)
        real(dp), intent(in) :: W_trial(:, :)

        integer :: n, i, j
        integer :: n_edge
        integer :: n_positive_reference, n_negative_reference
        integer :: n_positive_trial, n_negative_trial
        real(dp) :: sum_reference, sum_trial
        real(dp) :: sumsq_reference, sumsq_trial
        real(dp) :: min_reference, min_trial
        real(dp) :: max_reference, max_trial
        real(dp) :: sumabs_reference, sumabs_trial
        real(dp) :: scale_sum, scale_sumsq
        real(dp) :: tolerance_sum, tolerance_sumsq

        weight_shuffle_is_valid = .false.

        n = size(W_reference, 1)
        if (size(W_reference, 2) /= n) return
        if (size(W_trial, 1) /= n .or. size(W_trial, 2) /= n) return
        if (size(adj_matrix, 1) /= n .or. &
            size(adj_matrix, 2) /= n) return

        sum_reference = 0.0_dp
        sum_trial = 0.0_dp
        sumsq_reference = 0.0_dp
        sumsq_trial = 0.0_dp
        sumabs_reference = 0.0_dp
        sumabs_trial = 0.0_dp
        min_reference = huge(1.0_dp)
        min_trial = huge(1.0_dp)
        max_reference = -huge(1.0_dp)
        max_trial = -huge(1.0_dp)
        n_positive_reference = 0
        n_negative_reference = 0
        n_positive_trial = 0
        n_negative_trial = 0
        n_edge = 0

        do j = 1, n
            do i = 1, n
                if (.not. adj_matrix(i, j)) then
                    ! A non-edge must have zero weight before and after
                    ! shuffling, so topology cannot be altered.
                    if (W_reference(i, j) /= 0.0_dp .or. &
                        W_trial(i, j) /= 0.0_dp) return
                    cycle
                end if

                n_edge = n_edge + 1

                sum_reference = sum_reference + W_reference(i, j)
                sum_trial = sum_trial + W_trial(i, j)
                sumsq_reference = sumsq_reference + W_reference(i, j)**2
                sumsq_trial = sumsq_trial + W_trial(i, j)**2
                sumabs_reference = sumabs_reference + &
                    abs(W_reference(i, j))
                sumabs_trial = sumabs_trial + abs(W_trial(i, j))
                min_reference = min(min_reference, W_reference(i, j))
                min_trial = min(min_trial, W_trial(i, j))
                max_reference = max(max_reference, W_reference(i, j))
                max_trial = max(max_trial, W_trial(i, j))

                if (W_reference(i, j) > 0.0_dp) &
                    n_positive_reference = n_positive_reference + 1
                if (W_reference(i, j) < 0.0_dp) &
                    n_negative_reference = n_negative_reference + 1
                if (W_trial(i, j) > 0.0_dp) &
                    n_positive_trial = n_positive_trial + 1
                if (W_trial(i, j) < 0.0_dp) &
                    n_negative_trial = n_negative_trial + 1
            end do
        end do

        if (n_edge == 0) then
            weight_shuffle_is_valid = .true.
            return
        end if

        scale_sum = max(1.0_dp, &
            sumabs_reference, sumabs_trial)
        scale_sumsq = max(1.0_dp, &
            sumsq_reference, sumsq_trial)
        tolerance_sum = 64.0_dp * real(n_edge, dp) * &
            epsilon(1.0_dp) * scale_sum
        tolerance_sumsq = 64.0_dp * real(n_edge, dp) * &
            epsilon(1.0_dp) * scale_sumsq

        weight_shuffle_is_valid = &
            abs(sum_trial - sum_reference) <= tolerance_sum .and. &
            abs(sumsq_trial - sumsq_reference) <= tolerance_sumsq .and. &
            min_trial == min_reference .and. &
            max_trial == max_reference .and. &
            n_positive_trial == n_positive_reference .and. &
            n_negative_trial == n_negative_reference
    end function weight_shuffle_is_valid

    logical function value_distribution_is_preserved(reference, trial)
        real(dp), intent(in) :: reference(:)
        real(dp), intent(in) :: trial(:)

        integer :: n
        integer :: n_positive_reference, n_negative_reference
        integer :: n_positive_trial, n_negative_trial
        real(dp) :: sum_reference, sum_trial
        real(dp) :: sumsq_reference, sumsq_trial
        real(dp) :: sumabs_reference, sumabs_trial
        real(dp) :: scale_sum, scale_sumsq
        real(dp) :: tolerance_sum, tolerance_sumsq

        value_distribution_is_preserved = .false.

        n = size(reference)
        if (size(trial) /= n .or. n <= 0) return

        sum_reference = sum(reference)
        sum_trial = sum(trial)
        sumsq_reference = sum(reference**2)
        sumsq_trial = sum(trial**2)
        sumabs_reference = sum(abs(reference))
        sumabs_trial = sum(abs(trial))
        n_positive_reference = count(reference > 0.0_dp)
        n_negative_reference = count(reference < 0.0_dp)
        n_positive_trial = count(trial > 0.0_dp)
        n_negative_trial = count(trial < 0.0_dp)

        scale_sum = max(1.0_dp, &
            sumabs_reference, sumabs_trial)
        scale_sumsq = max(1.0_dp, &
            sumsq_reference, sumsq_trial)
        tolerance_sum = 64.0_dp * real(n, dp) * &
            epsilon(1.0_dp) * scale_sum
        tolerance_sumsq = 64.0_dp * real(n, dp) * &
            epsilon(1.0_dp) * scale_sumsq

        value_distribution_is_preserved = &
            abs(sum_trial - sum_reference) <= tolerance_sum .and. &
            abs(sumsq_trial - sumsq_reference) <= tolerance_sumsq .and. &
            minval(trial) == minval(reference) .and. &
            maxval(trial) == maxval(reference) .and. &
            n_positive_trial == n_positive_reference .and. &
            n_negative_trial == n_negative_reference
    end function value_distribution_is_preserved


end module shuffle_mod

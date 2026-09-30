program test_weight_shuffle_selection
    use precision_mod
    use random_mod, only : initialize_seed
    use network_mod, only : shuffle_fcnn_weights
    implicit none

    integer, parameter :: n = 6
    real(dp), parameter :: tol = 1.0e-14_dp
    logical :: adjacency(n, n)
    integer :: node_layer(n)
    real(dp) :: reference(n, n), trial(n, n)
    integer :: source, target, value, changed

    node_layer = [1, 1, 2, 2, 3, 3]
    adjacency = .false.
    reference = 0.0_dp
    value = 0
    do source = 1, 2
        do target = 3, 4
            value = value + 1
            adjacency(target, source) = .true.
            reference(target, source) = real(value, dp)
        end do
    end do
    do source = 3, 4
        do target = 5, 6
            value = value + 1
            adjacency(target, source) = .true.
            reference(target, source) = real(value, dp)
        end do
    end do

    ! A selected target layer means its incoming W_(ell,ell-1) block.
    call initialize_seed(721)
    trial = reference
    call shuffle_fcnn_weights( &
        adjacency, trial, node_layer, "LAYER", 3, 1.0_dp)
    if (any(trial(3:4, 1:2) /= reference(3:4, 1:2))) then
        error stop "Specific-layer shuffle changed an unselected block"
    end if
    call check_weight_multiset(reference, trial)
    call check_topology(adjacency, trial)

    ! A 50% global shuffle selects exactly four of the eight edge positions.
    call initialize_seed(722)
    trial = reference
    call shuffle_fcnn_weights( &
        adjacency, trial, node_layer, "GLOBAL", 0, 0.5_dp)
    changed = count(trial /= reference)
    if (changed <= 0 .or. changed > 4) then
        error stop "Partial global shuffle changed an invalid edge count"
    end if
    call check_weight_multiset(reference, trial)
    call check_topology(adjacency, trial)

    ! With LAYER scope the same fraction is sampled independently per block.
    call initialize_seed(723)
    trial = reference
    call shuffle_fcnn_weights( &
        adjacency, trial, node_layer, "LAYER", 0, 0.5_dp)
    changed = count(trial /= reference)
    ! Each block selects two positions, but a valid permutation may leave both
    ! selected values fixed. Therefore only the upper bound is invariant.
    if (changed > 4) then
        error stop "Partial layer shuffle changed an invalid edge count"
    end if
    if (abs(sum(trial(3:4, 1:2)**2) - &
            sum(reference(3:4, 1:2)**2)) > tol .or. &
        abs(sum(trial(5:6, 3:4)**2) - &
            sum(reference(5:6, 3:4)**2)) > tol) then
        error stop "Layer shuffle changed a block Frobenius norm"
    end if
    call check_weight_multiset(reference, trial)
    call check_topology(adjacency, trial)

    print *, "Selected-layer and partial weight shuffle tests passed."

contains

    subroutine check_weight_multiset(expected, actual)
        real(dp), intent(in) :: expected(:, :), actual(:, :)
        integer :: integer_weight

        do integer_weight = 1, 8
            if (count(actual == real(integer_weight, dp)) /= &
                count(expected == real(integer_weight, dp))) then
                error stop "Weight shuffle changed the weight multiset"
            end if
        end do
        if (abs(sum(actual**2) - sum(expected**2)) > tol) then
            error stop "Weight shuffle changed the global Frobenius norm"
        end if
    end subroutine check_weight_multiset

    subroutine check_topology(expected_adjacency, actual)
        logical, intent(in) :: expected_adjacency(:, :)
        real(dp), intent(in) :: actual(:, :)

        if (any(actual == 0.0_dp .neqv. .not. expected_adjacency)) then
            error stop "Weight shuffle changed the topology"
        end if
    end subroutine check_topology

end program test_weight_shuffle_selection

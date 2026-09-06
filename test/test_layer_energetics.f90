program test_layer_energetics
    use precision_mod
    use network_mod, only : assign_node_layers, &
        is_layered_feedforward, infer_fcnn_node_layers
    use output_mod, only : write_energetics_by_node_and_layer
    implicit none

    integer, parameter :: n = 5
    real(dp), parameter :: tolerance = 1.0e-13_dp
    character(len=*), parameter :: filename = &
        "test_node_and_layer_energetics.csv"
    logical :: adjacency(n, n)
    integer :: node_layer(n)
    integer :: layer_sizes(3)
    real(dp) :: heat(n), entropy(n), work(n), internal(n)
    integer :: io_unit, io_status, layer, node, expected_node
    integer, allocatable :: inferred_layer(:)
    logical :: is_fcnn
    real(dp) :: node_heat, node_entropy, node_work, node_internal
    character(len=256) :: line

    node_layer = [1, 1, 2, 2, 3]
    layer_sizes = [2, 2, 1]

    call assign_node_layers(layer_sizes, inferred_layer)
    if (any(inferred_layer /= node_layer)) then
        error stop "FCNN layer sizes were mapped incorrectly"
    end if

    adjacency = .false.
    adjacency(3, 1) = .true.
    adjacency(3, 2) = .true.
    adjacency(4, 1) = .true.
    adjacency(4, 2) = .true.
    adjacency(5, 3) = .true.
    adjacency(5, 4) = .true.

    if (.not. is_layered_feedforward(adjacency, node_layer)) then
        error stop "Valid layered FCNN topology was rejected"
    end if

    call infer_fcnn_node_layers(adjacency, inferred_layer, is_fcnn)
    if (.not. is_fcnn .or. any(inferred_layer /= node_layer)) then
        error stop "FCNN layers were inferred incorrectly"
    end if

    adjacency(5, 1) = .true.
    if (is_layered_feedforward(adjacency, node_layer)) then
        error stop "Non-adjacent FCNN edge was accepted"
    end if
    adjacency(5, 1) = .false.

    heat = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    entropy = 10.0_dp * heat
    work = 100.0_dp * heat
    internal = 1000.0_dp * heat

    call write_energetics_by_node_and_layer( &
        filename, node_layer, heat, work, internal, entropy)

    open(newunit=io_unit, file=filename, status="old", &
         action="read", iostat=io_status)
    if (io_status /= 0) error stop "Cannot read layer-output test file"

    read(io_unit, '(A)', iostat=io_status) line
    if (io_status /= 0) error stop "Missing layer-output title"
    read(io_unit, '(A)', iostat=io_status) line
    if (io_status /= 0) error stop "Missing layer-output header"

    do expected_node = 1, n
        read(io_unit, *, iostat=io_status) &
            layer, node, node_heat, node_entropy, &
            node_work, node_internal
        if (io_status /= 0) error stop "Missing node-layer output data"

        if (node /= expected_node .or. layer /= node_layer(expected_node)) then
            error stop "Incorrect node or layer ID"
        end if
        if (abs(node_heat - heat(expected_node)) > tolerance .or. &
            abs(node_entropy - entropy(expected_node)) > tolerance .or. &
            abs(node_work - work(expected_node)) > tolerance .or. &
            abs(node_internal - internal(expected_node)) > tolerance) then
            error stop "Incorrect node-layer energetic rates"
        end if
    end do

    close(io_unit, status="delete")

    print *, "Layer-wise theoretical energetics test passed."

end program test_layer_energetics

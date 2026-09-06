module network_mod
    use precision_mod
    use random_mod
    use parameter_mod
    implicit none

contains

    subroutine generate_er(param, adj_matrix, W)
        type(SimulationParameters), intent(in) :: param
        logical, allocatable, intent(out) :: adj_matrix(:,:)
        real(dp), allocatable, intent(out) :: W(:,:)
        integer :: i, j

        ! Allocate the adjacency matrix
        allocate(adj_matrix(param%N, param%N), W(param%N, param%N))
        adj_matrix = .false.
        W = 0.0_dp

        ! Generate ER graph
        if (param%directed) then ! Directed graph
            do i = 1, param%N
                do j = 1, param%N
                    if (i /= j) then
                        if (rand_uniform() < param%p) then
                            adj_matrix(i, j) = .true.
                            W(i, j) = param%weight_mean + param%weight_std * rand_normal()
                        end if
                    end if
                end do
            end do
        else ! Undirected graph
            do i = 1, param%N-1
                do j = i+1, param%N
                    if (rand_uniform() < param%p) then
                        adj_matrix(i, j) = .true.
                        adj_matrix(j, i) = .true.
                        W(i, j) = param%weight_mean + param%weight_std * rand_normal()
                        ! W(i, j) = -param%weight_mean + param%weight_std * rand_normal()
                        W(j, i) = W(i, j) ! Symmetric weights for undirected graph
                    end if
                end do
            end do
        end if

    end subroutine generate_er

    subroutine generate_ba()
    
    end subroutine generate_ba
    subroutine generate_ws

    end subroutine generate_ws

    subroutine generate_fcnn(param, n_hidden, layer_sizes, adj_matrix, W)
        type(SimulationParameters), intent(in) :: param
        integer, intent(in) :: n_hidden
        integer, intent(in) :: layer_sizes(:) ! no. of nodes in each layer
        logical, allocatable, intent(out) :: adj_matrix(:,:)
        real(dp), allocatable, intent(out) :: W(:,:)
        
        integer :: layer
        integer :: source_node, target_node
        integer :: source_first, source_last
        integer :: target_first, target_last
        integer :: n_layers, n_total
        integer, allocatable :: offset(:)
        real(dp) :: weight

        !-----------------------------------------
        ! 1. 檢查輸入
        !-----------------------------------------

        if (n_hidden < 0) then
            error stop "n_hidden must be nonnegative"
        end if

        n_layers = n_hidden + 2

        if (size(layer_sizes) /= n_layers) then
            error stop "layer_sizes must contain input, hidden, and output layers"
        end if

        if (any(layer_sizes <= 0)) then
            error stop "Every layer must contain at least one node"
        end if

        n_total = sum(layer_sizes) ! total no. of nodes

        if (param%N /= n_total) then
            error stop "param%N does not match sum(layer_sizes)"
        end if

        !-----------------------------------------
        ! 2. 配置與初始化矩陣
        !-----------------------------------------

        allocate(adj_matrix(n_total, n_total))
        allocate(W(n_total, n_total))

        adj_matrix = .false.
        W = 0.0_dp

        !-----------------------------------------
        ! 3. 計算每層的 offset (offset(n) = no. of nodes before n^th layer)
        !-----------------------------------------

        allocate(offset(n_layers)); offset(1) = 0

        do layer = 2, n_layers
            offset(layer) = offset(layer-1) + layer_sizes(layer-1)
        end do

        !-----------------------------------------
        ! 4. 連接相鄰兩層
        !-----------------------------------------

        do layer = 1, n_layers - 1

            source_first = offset(layer) + 1 ! the first node in the n^th layer
            source_last  = offset(layer) + layer_sizes(layer) ! the last node in the n^th layer

            target_first = offset(layer+1) + 1 ! the first node in the n^th layer
            target_last  = offset(layer+1) + layer_sizes(layer+1) ! the last node in the n^th layer

            do source_node = source_first, source_last
                do target_node = target_first, target_last

                    weight = param%weight_mean + param%weight_std * rand_normal()

                    ! source -> target
                    adj_matrix(target_node, source_node) = .true.
                    W(target_node, source_node) = weight

                    ! ! 如果需要 undirected layered network
                    ! if (.not. param%directed) then
                    !     adj_matrix(source_node, target_node) = .true.
                    !     W(source_node, target_node) = weight
                    ! end if

                enddo
            enddo

        enddo

    end subroutine generate_fcnn

    subroutine shuffle_FCNN_weights(adj_matrix, W)
        logical, intent(in) :: adj_matrix(:,:)
        real(dp), intent(inout) :: W(:,:)

        integer :: n, n_edges
        integer :: i, j, k, random_index, edge_index
        real(dp) :: temp
        real(dp), allocatable :: edge_weights(:)

        n = size(W, 1)

        if (size(W, 2) /= n) then
            error stop "W must be square"
        end if

        if (size(adj_matrix, 1) /= n .or. &
            size(adj_matrix, 2) /= n) then
            error stop "adj_matrix and W size mismatch"
        end if

        n_edges = count(adj_matrix)

        if (n_edges <= 1) return

        allocate(edge_weights(n_edges))

        ! 收集所有既有 edge 的權重
        edge_index = 0

        do j = 1, n
            do i = 1, n
                if (adj_matrix(i, j)) then
                    edge_index = edge_index + 1
                    edge_weights(edge_index) = W(i, j)
                end if
            end do
        end do

        ! Fisher-Yates shuffle
        do k = n_edges, 2, -1
            random_index = 1 + int(rand_uniform() * real(k, dp))

            temp = edge_weights(k)
            edge_weights(k) = edge_weights(random_index)
            edge_weights(random_index) = temp
        end do

        ! 將排列後的權重放回相同的 topology
        edge_index = 0

        do j = 1, n
            do i = 1, n
                if (adj_matrix(i, j)) then
                    edge_index = edge_index + 1
                    W(i, j) = edge_weights(edge_index)
                end if
            end do
        end do

    end subroutine shuffle_FCNN_weights

    subroutine read_weighted_edge_list(filename, adj_matrix, W, n_nodes)
        use, intrinsic :: iso_fortran_env, only : iostat_end

        character(len=*), intent(in) :: filename
        logical, allocatable, intent(out) :: adj_matrix(:,:)
        real(dp), allocatable, intent(out) :: W(:,:)
        integer, intent(out) :: n_nodes

        character(len=1024) :: line
        integer :: io_unit
        integer :: io_status, parse_status
        integer :: line_number
        integer :: i, j, n_edges
        real(dp) :: wij        

        open(newunit=io_unit, file=trim(filename), & ! 移除右邊空白
         status="old", action="read", iostat=io_status)

        if (io_status /= 0) then
            error stop "Cannot open weighted edge-list file"
        end if

        n_nodes = 0
        n_edges = 0
        line_number = 0

        do
            read(io_unit, '(A)', iostat=io_status) line
            if (io_status == iostat_end) exit
            if (io_status /= 0) then
                error stop "Error reading weighted edge-list file"
            end if

            line_number = line_number + 1
            line = adjustl(line) ! 移除左邊空白

            if (len_trim(line) == 0) cycle
            if (line(1:1) == "#" .or. line(1:1) == "!") cycle

            read(line, *, iostat=parse_status) i, j, wij

            if (parse_status /= 0) then
                error stop "Invalid weighted edge-list record"
            end if
            if (i <= 0 .or. j <= 0) then
                error stop "Node indices must start from 1"
            end if
            if (i == j) then
                error stop "Self-loops are not supported"
            end if
            n_nodes = max(n_nodes, i, j)
            n_edges = n_edges + 1
        end do

        if (n_edges == 0) then
            error stop "Weighted edge-list file is empty"
        end if

        allocate(adj_matrix(n_nodes,n_nodes))
        allocate(W(n_nodes,n_nodes))

        adj_matrix = .false.
        W = 0.0_dp

        rewind(io_unit)

        do
            read(io_unit, '(A)', iostat=io_status) line
            if (io_status == iostat_end) exit
            if (io_status /= 0) then
                error stop "Error reading weighted edge-list file"
            end if

            line = adjustl(line)

            if (len_trim(line) == 0) cycle
            if (line(1:1) == "#" .or. line(1:1) == "!") cycle

            read(line, *, iostat=parse_status) i, j, wij

            if (parse_status /= 0) then
                error stop "Invalid weighted edge-list record"
            end if

            ! 重複 edge 不應默默覆蓋
            if (adj_matrix(i,j)) then
                error stop "Duplicate edge in weighted edge-list file"
            end if

            ! j -> i
            adj_matrix(i,j) = .true.
            W(i,j) = wij
        end do

        close(io_unit)

    end subroutine read_weighted_edge_list                          

    subroutine assign_node_layers(layer_sizes, node_layer)
        ! Convert FCNN layer sizes into one zero-based layer ID per node.
        integer, intent(in) :: layer_sizes(:)
        integer, allocatable, intent(out) :: node_layer(:)
        integer :: layer, first_node, last_node, n_nodes

        if (size(layer_sizes) <= 0) then
            error stop "layer_sizes must not be empty"
        end if
        if (any(layer_sizes <= 0)) then
            error stop "Every layer must contain at least one node"
        end if

        n_nodes = sum(layer_sizes)
        allocate(node_layer(n_nodes))

        first_node = 1
        do layer = 1, size(layer_sizes)
            last_node = first_node + layer_sizes(layer) - 1
            node_layer(first_node:last_node) = layer - 1
            first_node = last_node + 1
        end do

    end subroutine assign_node_layers

    logical function is_layered_feedforward(adj_matrix, node_layer)
        ! A layered FCNN edge must connect layer l to layer l+1.
        logical, intent(in) :: adj_matrix(:,:)
        integer, intent(in) :: node_layer(:)
        integer :: n, target, source, layer

        n = size(node_layer)
        is_layered_feedforward = .false.

        if (n <= 0) return
        if (size(adj_matrix, 1) /= n .or. size(adj_matrix, 2) /= n) return
        if (any(node_layer < 0)) return
        if (maxval(node_layer) <= 0) return

        ! Layer IDs must be contiguous from the input layer 0 onward.
        do layer = 0, maxval(node_layer)
            if (count(node_layer == layer) == 0) return
        end do

        do source = 1, n
            do target = 1, n
                if (.not. adj_matrix(target, source)) cycle
                if (node_layer(target) /= node_layer(source) + 1) return
            end do
        end do

        is_layered_feedforward = .true.

    end function is_layered_feedforward

    subroutine infer_fcnn_node_layers( &
        adj_matrix, node_layer, is_fcnn)
        ! Infer zero-based layers from a directed, adjacent-layer FCNN.
        logical, intent(in) :: adj_matrix(:,:)
        integer, allocatable, intent(out) :: node_layer(:)
        logical, intent(out) :: is_fcnn
        integer, allocatable :: remaining_indegree(:)
        logical, allocatable :: current_layer_nodes(:)
        integer :: n, target, source, layer
        integer :: n_assigned, expected_edges, actual_edges

        n = size(adj_matrix, 1)
        is_fcnn = .false.

        if (n <= 0 .or. size(adj_matrix, 2) /= n) then
            allocate(node_layer(0))
            return
        end if

        allocate(node_layer(n), remaining_indegree(n))
        allocate(current_layer_nodes(n))

        node_layer = -1
        do target = 1, n
            remaining_indegree(target) = count(adj_matrix(target, :))
        end do

        layer = 0
        n_assigned = 0

        do while (n_assigned < n)
            current_layer_nodes = &
                node_layer < 0 .and. remaining_indegree == 0

            if (count(current_layer_nodes) == 0) return

            where (current_layer_nodes)
                node_layer = layer
            end where
            n_assigned = n_assigned + count(current_layer_nodes)

            do source = 1, n
                if (.not. current_layer_nodes(source)) cycle
                do target = 1, n
                    if (adj_matrix(target, source)) then
                        remaining_indegree(target) = &
                            remaining_indegree(target) - 1
                    end if
                end do
            end do

            layer = layer + 1
        end do

        if (.not. is_layered_feedforward(adj_matrix, node_layer)) return

        ! A full FCNN contains every edge between each adjacent layer pair.
        do layer = 0, maxval(node_layer) - 1
            expected_edges = count(node_layer == layer) * &
                count(node_layer == layer + 1)
            actual_edges = 0

            do source = 1, n
                if (node_layer(source) /= layer) cycle
                do target = 1, n
                    if (node_layer(target) /= layer + 1) cycle
                    if (adj_matrix(target, source)) then
                        actual_edges = actual_edges + 1
                    end if
                end do
            end do

            if (actual_edges /= expected_edges) return
        end do

        is_fcnn = .true.

    end subroutine infer_fcnn_node_layers

end module network_mod

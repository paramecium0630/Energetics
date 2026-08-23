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
                        ! W(i, j) = param%weight_mean + param%weight_std * rand_normal()
                        W(i, j) = -param%weight_mean + param%weight_std * rand_normal()
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

                    weight = param%weight_mean

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

end module network_mod
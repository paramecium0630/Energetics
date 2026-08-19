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
    subroutine generate_fcnn()

    end subroutine generate_fcnn

end module network_mod
module alignment_mod
        
        use, intrinsic :: iso_c_binding
    implicit none

    interface
        function aligned_alloc_16(size) bind(C, name="aligned_alloc_16")
            import :: c_size_t, c_ptr
            integer(c_size_t), value :: size
            type(c_ptr) :: aligned_alloc_16
        end function aligned_alloc_16

        subroutine aligned_free(ptr) bind(C, name="aligned_free")
            import :: c_ptr
            type(c_ptr), value :: ptr
        end subroutine aligned_free
    end interface

end module

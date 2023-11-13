! load.f90

module mixed_type_module
  use precision_module
  implicit none
  type mixed_type
    class(*), allocatable :: item
  end type mixed_type

  type multi_type
        integer :: type_num
        integer(4) :: i32
        !integer(2) :: i16
        real(4) :: f32
        character(64)  :: string
        type(multi_type), allocatable :: a(:)
  end type

  type ggml_tensor_info
        character(64) :: tname
        integer(4) :: ndim, ttype
        integer(8) :: offset
        integer(8), allocatable :: dims(:)
  end type

  type generic_tensor
        integer :: ndims
        integer :: ttype
        integer(2), allocatable :: f161d(:)
        integer(2), allocatable :: f162d(:,:)
        real(kind=wp), allocatable :: f321d(:)
        real(kind=wp), allocatable :: f322d(:,:)
        ! can add fp4
  end type


end module


module read_ggml

        use precision_module
        use mixed_type_module
        use weight_module
        implicit none
        
        type(ggml_tensor_info), allocatable :: tensors(:)
        logical :: verbose
        !integer :: file_pos
        integer(8) :: tensor_count
        logical, parameter :: verbose2 = .false.
contains
        subroutine load_ggml(filename, w, c, vocab, scores, token_lengths, v)
        character(len=*), intent(in) :: filename
        type(TransformerWeights), intent(out) :: w
        type(Config), intent(out) :: c
        real(kind=wp), allocatable, intent(out) :: scores(:)
        character(:), dimension(:), allocatable, intent(out) :: vocab
        integer(4), allocatable, intent(out) :: token_lengths(:)
        logical, intent(in) :: v
        
        character(:), dimension(:), allocatable :: vocab_swp
        integer(4) :: magic, version
        integer(8) :: kv_pairs
        !class(*), allocatable :: demo
        integer :: max_len = 64
        integer :: i, j, val_type,file_pos,  alignment, deficit 
        integer(4) :: num_layers, emb_length, context_length, head_count, ffn_length, kv_heads, vocab_size
        type(multi_type), allocatable :: values(:)
        type(multi_type) :: multi_temp
        character(:), dimension(:), allocatable :: keys
        !type(multi_type), allocatable :: x(:) 
        !type(ggml_tensor_info), allocatable :: tensors(:)
        type(ggml_tensor_info) :: t0
        !demo = 3
        integer(1) :: tbyte
        integer(1) :: tbytes(3)
        integer(2) :: f16
        integer(2), allocatable :: temp2f16(:,:)
        integer(2), allocatable :: tempf16(:)
        real(kind=wp), allocatable :: tempf32(:)
        real(kind=wp), allocatable :: temp2f32(:,:)
        character(:), allocatable :: tempstr
        type(generic_tensor) :: temp_gt
        !type (args) :: arg_values


        !real(kind=wp), allocatable :: scores(:)
        !character(:), dimension(:), allocatable :: vocab
        !integer(4), allocatable :: token_lengths(:)
        integer(8) :: tmp_vocab_size
        integer(4) :: temp_int, maxlen

        integer(8) :: strlen

        character(:), allocatable :: loaded_str
        
        integer :: head_size, kv_head_size
        
        allocate(character(len=max_len) :: tempstr)
        verbose = v
        
        ! assumed to be 32 if not specified
        alignment = 32
        num_layers = 0

        open(UNIT=5, FILE=filename, FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
                
                ! config
        
                read(5) magic, version, tensor_count, kv_pairs

                if (verbose) then
                        print *, "GGUF Header Info"
                        print *, "Magic number: ", magic
                        print *, "Version: ", version
                        print *, "Tensor Count: ", tensor_count
                        print *, "Key-Value Pairs: ", kv_pairs
                end if

                if (magic .ne. 1179993927) then
                        print *, "Magic numbers do not match, exiting"
                        stop
                end if

                allocate(character(len=max_len) ::  keys(kv_pairs))
                allocate(values(kv_pairs))
                do i = 1,kv_pairs
                        keys(i) = read_str(5)
                        read(5) val_type
                        values(i) = read_val(5,val_type)
                        if (keys(i) .eq. "general.alignment") then
                                alignment = values(i)%i32
                                if (verbose) then 
                                        print *, "alignment set to", alignment
                                end if
                        else if (keys(i) .eq. "llama.block_count") then
                                num_layers = values(i)%i32 !assume it's int(4)
                        else if (keys(i) .eq. "llama.embedding_length") then
                                emb_length = values(i)%i32
                        else if (keys(i) .eq. "llama.attention.head_count") then
                                head_count = values(i)%i32
                        else if (keys(i) .eq. "llama.context_length") then
                                context_length = values(i)%i32
                        else if (keys(i) .eq. "tokenizer.ggml.tokens") then
                                vocab_size = (size(values(i)%a))
                        else if (keys(i) .eq. "llama.attention.head_count_kv") then
                                kv_heads = values(i)%i32
                        else if (keys(i) .eq. "llama.feed_forward_length") then
                                ffn_length = values(i)%i32
                        end if
                        
                        if (verbose) then
                        print *, keys(i)
                        call print_multi(values(i))
                        end if
                end do

                allocate(tensors(tensor_count))
                do i = 1,tensor_count
                        tensors(i) = read_tensor_info(5)
                end do

                ! "level 2 verbose"
                if (verbose2) then
                        do i = 1, tensor_count
                                write (*, fmt="(A20,I2)",advance="no") tensors(i)%tname, tensors(i)%ndim
                                do j=1,tensors(i)%ndim
                                write (*, fmt="(I6)", advance="no") tensors(i)%dims(j) 
                                end do        
                                write (*, fmt="(I2,I11)") tensors(i)%ttype, tensors(i)%offset 
                        end do
                end if
                
                inquire(unit=5,pos=file_pos)

                deficit = mod(file_pos-1,alignment) ! -1

                if (verbose) then
                print *, "Position", file_pos
                print *, "Deficit", deficit
                end if

                if (deficit > 0) then
                do i = 1,(alignment-deficit)
                        read (5) tbyte
                        if (tbyte /= 0) then
                                print *, "padding error", tbyte
                        end if
                end do
        end if

                inquire(unit=5,pos=file_pos)

                print *, "data offset", file_pos

                !read(5) f16
               
                !print *, "First value", half_to_float_c(f16) 

              
                !if (outfile /= "") then
                !open(unit=8, file=outfile, form='unformatted', status='unknown', ACCESS="STREAM", action="write")
                ! write the header:
                if (verbose) then 
                        if (verbose) then
                        print *, "Embedding dimension: ", emb_length
                        print *, "Hidden dimension: ", ffn_length
                        print *, "Layers: ", num_layers
                        print *, "Heads: ", head_count
                        print *, "kv Heads: ", kv_heads
                        print *, "Vocabulary Size: ", vocab_size
                        print *, "Sequence Length: ", context_length

                end if

                        !print *, "Header:"
                        !print *, emb_length, ffn_length, num_layers, head_count, kv_heads, vocab_size, context_length
                end if 
                !write(8) emb_length, ffn_length, num_layers, head_count, kv_heads, vocab_size, context_length
                c%emb_dim = emb_length
                c%hidden_dim = ffn_length
                c%n_layers = num_layers
                c%n_heads = head_count
                c%n_kv_heads = kv_heads
                c%vocab_size = vocab_size
                c%seq_len = context_length 
                
                head_size = emb_length / head_count
                kv_head_size = kv_heads * head_size
                
                if (verbose) then
                print *, "head size ", head_size
                print *, "kv head Size ", kv_head_size
                end if

                t0 = tensor_by_name("token_embd.weight")
                temp_gt = read_layer(5,t0,file_pos)

                !call write_tensor(8,temp_gt)
                w%token_embedding_table = temp_gt%f322d

                if (verbose) then
                        print *, "loaded embedding weights:", size(w%token_embedding_table)
                end if
                !print *, temp_gt%ttype
                !print *, temp_gt%ndims
                !print *, w%token_embedding_table(1:10,1)
                !print *, "embed sum: ", sum(w%token_embedding_table(1:10,1:10))

                allocate(w%rms_att_weight(emb_length,num_layers))
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".attn_norm.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! should be f32
                        !call write_tensor(8,temp_gt)
                        w%rms_att_weight(:,i) = temp_gt%f321d
                end do
                if (verbose) then
                        print *, "loaded rms att weights:", size(w%rms_att_weight)
                end if

                allocate(w%wqkv(emb_length,emb_length+2*kv_head_size,num_layers))
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".attn_q.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16
                        !call write_tensor(8,temp_gt)
                        w%wqkv(:,1:emb_length,i) = temp_gt%f322d
                end do 

                if (verbose) then
                        print *, "loaded wq weights:", size(w%wqkv(:,1:emb_length,:))
                end if

               
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".attn_k.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16 
                        !call write_tensor(8,temp_gt)
                        w%wqkv(:,(emb_length+1):(emb_length+kv_head_size),i) = temp_gt%f322d
                end do

                if (verbose) then
                        print *, "loaded wk weights:", size(w%wqkv(:,(emb_length+1):(emb_length+kv_head_size),:))
                end if


                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".attn_v.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16
                        !call write_tensor(8,temp_gt)
                        w%wqkv(:,(emb_length+kv_head_size+1):(emb_length+2*kv_head_size),i) = temp_gt%f322d
                end do

                !print *, "qkv sum: ", sum(w%wqkv)
                if (verbose) then
                        print *, "loaded wv weights:", size(w%wqkv(:,(emb_length+kv_head_size+1):,:))
                end if



                allocate(w%wo(emb_length,emb_length,num_layers))
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".attn_output.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16
                        !call write_tensor(8,temp_gt)
                        w%wo(:,:,i) = temp_gt%f322d
                end do

                if (verbose) then
                        print *, "loaded wo weights:", size(w%wo)
                end if


                allocate(w%rms_ffn_weight(emb_length,num_layers))
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".ffn_norm.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f32
                        !call write_tensor(8,temp_gt)
                        w%rms_ffn_weight(:,i) = temp_gt%f321d
                end do

                if (verbose) then
                        print *, "loaded ffn norm weights:", size(w%rms_ffn_weight)
                end if


                allocate(w%w13(emb_length,2*ffn_length,num_layers))
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".ffn_gate.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16
                        !call write_tensor(8,temp_gt)
                        w%w13(:,1:ffn_length,i) = temp_gt%f322d
                end do

                if (verbose) then
                        print *, "loaded w1 (gate) weights:", size(w%w13(:,1:ffn_length,:))
                end if


                allocate(w%w2(ffn_length,emb_length,num_layers))
                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".ffn_down.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16
                        !call write_tensor(8,temp_gt)
                        w%w2(:,:,i) = temp_gt%f322d
                end do

                if (verbose) then
                        print *, "loaded w2 (down) weights:", size(w%w2)
                end if


                do i = 1,num_layers
                        write(tempstr,"(A,I0,A)") "blk.", i-1, ".ffn_up.weight"
                        t0 = tensor_by_name(tempstr)
                        temp_gt = read_layer(5,t0,file_pos)
                        ! f16
                        !call write_tensor(8,temp_gt)
                        w%w13(:,(ffn_length+1):,i) = temp_gt%f322d
                end do

                if (verbose) then
                        print *, "loaded w3 (up) weights:", size(w%w13(:,(ffn_length+1):,:))
                end if


                t0 = tensor_by_name("output_norm.weight")
                temp_gt = read_layer(5,t0,file_pos)
                ! f32
                !call write_tensor(8,temp_gt)
                w%rms_final_weight = temp_gt%f321d

                if (verbose) then
                        print *, "loaded output norm weights:", size(w%rms_final_weight)
                end if


                !temp2f32 = get_rope_freqs(emb_length/head_count,context_length,10000.0)
                !if (verbose) then
                !write(*,"(A)") "rope cos: writing float32"
                !end if
                !write(8) cos(temp2f32(:,:context_length))
                !if (verbose) then
                !write(*,"(A)") "rope sin: writing float32"
                !end if
                !write(8) sin(temp2f32(:,:context_length))
                ! cos and sin of the above are the cos/sin respectively (f32)

                t0 = tensor_by_name("output.weight")
                temp_gt = read_layer(5,t0,file_pos)
                ! f16
                !call write_tensor(8,temp_gt)
                w%wcls = temp_gt%f322d

                if (verbose) then
                        print *, "loaded classifier weights:", size(w%wcls)
                end if


        !close(8)
        !end if ! writing outfile 

        if (.true.) then
                ! just read and write the values again:
                call fseek(5,0,0) 
                read(5) magic, version, tensor_count, kv_pairs

                if (magic .ne. 1179993927) then
                        print *, "Magic numbers do not match, exiting"
                        stop
                end if

                do i = 1,kv_pairs
                        tempstr = read_str(5)
                        read(5) val_type
                        if (verbose2) then
                        print *, "scanning ", tempstr
                        end if
                        if (tempstr .eq. "tokenizer.ggml.tokens") then
                                if (verbose) then
                                print *, "loading tokens"
                                end if
                                ! allocate
                                read(5) temp_int, tmp_vocab_size
                        !allocate(val%a(alen))
                        !do i = 1,alen
                        ! val%a(i) = read_val(handle, atype)
                        !end do
                                allocate(character(len=max_len) ::  vocab(tmp_vocab_size))
                                allocate(token_lengths(tmp_vocab_size))
                                do j=1,int(tmp_vocab_size,4)
                                        read(5) strlen
                                        allocate(character(strlen) :: loaded_str)
                                        read(5) loaded_str
                                        token_lengths(j) = int(strlen,4)
                                        vocab(j) = loaded_str
                                        deallocate(loaded_str)
                                end do
                                if (verbose) then
                                write (*,"(A,I0,A)") "found ", size(vocab), " tokens"
                                end if
        
                        else if (tempstr .eq. "tokenizer.ggml.scores") then
                                multi_temp = read_val(5,val_type)
                                allocate(scores(size(multi_temp%a)))
                                do j = 1,size(multi_temp%a)
                                        scores(j) = multi_temp%a(j)%f32
                                end do
                                if (verbose) then
                                write (*,"(A,I0,A)") "found ", size(multi_temp%a), " scores" 
                                end if
                        else
                        multi_temp = read_val(5,val_type)
                        end if
                end do

                !open(unit=8, file="", form='unformatted', status='unknown', ACCESS="STREAM", action="write")
                maxlen = maxval(token_lengths)
                
                allocate(character(len=max_len) ::  vocab_swp(tmp_vocab_size))
                if (verbose) then
                print *, "maximum token length ", maxlen
                end if
                !temp_int = 10
                !write(8) maxlen 
                do i=1,size(vocab)
                read(vocab(i)(1:1), "(A)") tbytes(1)
                read(vocab(i)(2:2), "(A)") tbytes(2)
                read(vocab(i)(3:3), "(A)") tbytes(3)

                !end if
                if ( (tbytes(1) .eq. -30) .and.&
                        &(tbytes(2) .eq. -106) .and.&
                        &(tbytes(3) .eq. -127) ) then
                allocate(character(token_lengths(i)-2) :: loaded_str)
                loaded_str(1:1) = " "
                loaded_str(2:) = vocab(i)(4:token_lengths(i))
                !write(8) scores(i),token_lengths(i)-2,loaded_str
                token_lengths(i) = token_lengths(i)-2
                vocab_swp(i) = loaded_str
                deallocate(loaded_str)
                else
                !write(8) scores(i),token_lengths(i),vocab(i)(1:token_lengths(i))
                vocab_swp(i) = vocab(i)(1:token_lengths(i))
        end if
                end do

        end if

        !close(8)

        close(5)
        vocab = vocab_swp
        end subroutine

        
        subroutine write_tensor(handle, t)
                integer :: handle
                type(generic_tensor) :: t
                
                if (t%ttype .eq. 0) then
                        if (verbose) then
                                write(*,"(A)") "writing float32"
                        end if
                        if (t%ndims .eq. 1) then
                                write(handle) t%f321d
                        else if (t%ndims .eq. 2) then
                                write(handle) t%f322d
                        end if
                else if (t%ttype .eq. 1) then
                        if (verbose) then
                                write(*,"(A)") "writing fp16"
                        end if
                        if (t%ndims .eq. 1) then
                                write(handle) t%f161d
                        else if (t%ndims .eq. 2) then
                                write(handle) t%f162d
                        end if
                end if


        end subroutine 
        
        function get_rope_freqs(i_dim, i_end, theta) result(freq_array)
                integer :: i_dim, i_end
                real(kind=wp) :: theta
                !real(kind=wp) :: cis(i_end/2,2)
                real(kind=wp),allocatable :: freqs(:)
                real(kind=wp),allocatable :: freq_array(:,:)
                real(kind=wp) :: irange(i_dim/2)
                integer :: i
                do i = 1,i_dim/2
                        irange(i) = 2.0*(i-1) / i_dim
                        freqs = 1.0 / (theta ** irange) 
                
                end do
                allocate(freq_array(size(freqs),i_end)) ! may need transposing
                do i = 0,(i_end-1)
                freq_array(:,i+1) = i*freqs
                end do

        end function
        
        function tensor_by_name(s)
                character(len=*) :: s
                integer :: i
                type(ggml_tensor_info) :: tensor_by_name
                do i=1,tensor_count
                        if (tensors(i)%tname .eq. s) then
                                tensor_by_name = tensors(i)
                                return 
                        end if
                end do 
                print *, "key not found",s
                stop
        end
        function prod(a)
                integer(8) :: a(:)
                integer :: i
                integer(8) :: prod
                prod = 1
                do i = 1,size(a)
                 prod = prod * a(i)
                 end do 
        end function
    
        function read_layer_fp16(handle, layer) result(d)
                integer :: handle
                type(ggml_tensor_info) :: layer
                integer(2), allocatable :: d(:)
                if (verbose) then
                        write(*,"(A,A26)",advance="no") "reading",layer%tname 
                end if
                !call fseek(handle,layer%offset+file_pos,0)
                allocate(d(prod(layer%dims)))
                read(handle) d
                if (verbose) then
                        write(*,"(A)") "... done"
                end if 

        end function 

        function read_layer(handle, layer,file_pos) result(d)
                integer :: handle
                type(ggml_tensor_info) :: layer
                type(generic_tensor) :: d
                integer :: file_pos
                !integer(2), allocatable :: d(:)
                !if (verbose) then
                !        write(*,"(A,A26)",advance="no") "reading",layer%tname
                !end if
                call fseek(handle,layer%offset+file_pos-1,0)
                d%ttype = layer%ttype
                d%ndims = layer%ndim
                
                if (d%ttype .eq. 0) then
                        if (d%ndims .eq. 1) then
                                allocate(d%f321d(layer%dims(1)))
                                read(handle) d%f321d
                        else if (d%ndims .eq. 2) then
                                allocate(d%f322d(layer%dims(1),layer%dims(2)))
                                read(handle) d%f322d
                        else
                        print *, "Ndims nuot supported", layer%dims
                        end if
                else if (d%ttype .eq. 1) then
                        if (d%ndims .eq. 1) then
                                allocate(d%f161d(layer%dims(1)))
                                read(handle) d%f161d
                        else if (d%ndims .eq. 2) then
                                allocate(d%f162d(layer%dims(1),layer%dims(2)))
                                read(handle) d%f162d
                        else
                        print *, "Ndims not supported", layer%dims
                        end if
                else
                        print *, "Type not supported", layer%ttype
                end if
                
                !if (verbose) then
                !        write(*,"(A)") "... done"
                !end if

        end function   
        
        function read_str(handle)
                integer :: handle
                integer(8) :: strlen
           
                character(:), allocatable :: read_str
                read(handle) strlen
                allocate(character(strlen) :: read_str)
                read(handle) read_str
                
        end function

        recursive function read_val(handle, val_type) result (val)
                integer :: handle, val_type, i
                character (:), allocatable :: temp
                type(multi_type) :: val
                integer(4) :: atype
                integer(8) :: alen

                val%type_num = val_type
                
                if (val_type .eq. 8) then
                        temp = read_str(handle)
                        !print *, temp
                        val%string = temp

                else if (val_type .eq. 4) then
                        ! read in an int32
                        read(handle) val%i32
                else if (val_type .eq. 6) then
                        read(handle) val%f32 
                else if (val_type .eq. 5) then
                        read(handle) val%i32 
                else if (val_type .eq. 9) then
                        read(handle) atype, alen
                        allocate(val%a(alen))
                        do i = 1,alen
                         val%a(i) = read_val(handle, atype)
                        end do

                else 
                        print *, "Not implemented", val_type
                        stop
                end if


        end function

        subroutine print_multi(m) 
                type(multi_type) :: m
                if (m%type_num .eq. 8) then
                        print *, m%string
                else if (m%type_num .eq. 4) then
                       print *, m%i32 
                else if (m%type_num .eq. 5) then
                        print *, m%i32
                else if (m%type_num .eq. 6) then
                       print *, m%f32         
               else if (m%type_num .eq. 9) then
                       print *, size(m%a)
                end if

        end subroutine

        function read_tensor_info(handle) result(info)
                integer :: handle, i
                type(ggml_tensor_info) :: info
                info%tname = read_str(handle)
                read(handle) info%ndim 
                allocate(info%dims(info%ndim))
                do i = 1,info%ndim
                        read(handle) info%dims(i)
                end do
                read(handle) info%ttype
                read(handle) info%offset

        end function
        

end module

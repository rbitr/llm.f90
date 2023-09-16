! llama2.f90
! implementation of a toy llama model

! set precision of reals (here default which is 4 byte)
module precision_module
  implicit none
  integer, parameter :: wp = kind(1.0)
end module precision_module

! structs for reading weights, config information and state 
module weight_module
        use precision_module
        implicit none
        private wp

        type TransformerWeights
                integer(2), allocatable :: token_embedding_table(:,:)
                real(kind=wp), allocatable :: rms_att_weight(:,:)
                real(kind=wp), allocatable :: rms_ffn_weight(:,:)
                integer(2), allocatable :: wq(:,:,:)
                integer(2), allocatable :: wk(:,:,:)
                integer(2), allocatable :: wv(:,:,:)
                integer(2), allocatable :: wo(:,:,:)
                integer(2), allocatable :: w1(:,:,:)
                integer(2), allocatable :: w2(:,:,:)
                integer(2), allocatable :: w3(:,:,:)
                real(kind=wp), allocatable :: rms_final_weight(:)
                real(kind=wp), allocatable :: freq_cis_real(:,:)
                real(kind=wp), allocatable :: freq_cis_imag(:,:)
                integer(2), allocatable :: wcls(:,:)

        end type TransformerWeights

        type Config
                INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        end type Config

        type RunState

                real(kind=wp), allocatable :: att(:,:)
                real(kind=wp), allocatable :: key_cache(:,:,:)
                real(kind=wp), allocatable :: value_cache(:,:,:)
                real(kind=wp) :: times(5)

        end type RunState

end module weight_module

module arg_parse
        implicit none

        type args
                real :: temperature
                character(:), allocatable :: model_file
                character(:), allocatable :: prompt
                character(:), allocatable :: tokenizer
                logical :: verbose
                integer :: n
        end type args

        contains

                subroutine parse_args(arg_values)
                        type(args) :: arg_values 
                        integer :: i, num_args
                        character(256) :: arg

                        

                        !defaults 
                        arg_values%temperature = 0
                        arg_values%model_file = "stories15M.bin"
                        arg_values%prompt = ""
                        arg_values%verbose = .false.
                        arg_values%n = 256
                        arg_values%tokenizer = "tokenizer.bin"
                
                        num_args = command_argument_count()

                        i = 1
                        do while (i <= num_args)
                                call get_command_argument(i, arg)
                                        select case (arg)
                                                case ('-m', '--model')
                                                ! path to model file
                                                call get_command_argument(i+1, arg)
                                                arg_values%model_file = trim(arg)
                                                i = i + 2
                                                case ('-p', '--prompt')
                                                ! prompt string
                                                call get_command_argument(i+1, arg)
                                                arg_values%prompt = trim(arg)
                                                i = i + 2
                                                case ('-s', '--tokenizer')
                                                ! path to custom tokenizer
                                                call get_command_argument(i+1, arg)
                                                arg_values%tokenizer = trim(arg)
                                                i = i + 2
                                                case ('-t', '--temperature')
                                                ! temperature scaling
                                                call get_command_argument(i+1, arg)
                                                read(arg,*) arg_values%temperature
                                                i = i + 2
                                                case ('-n', '--num_tokens')
                                                ! number of tokens to generate, including prompt
                                                call get_command_argument(i+1, arg)
                                                read(arg,*) arg_values%n
                                                i = i + 2
                                                case ('-v', '--verbose')
                                                ! print additional information
                                                arg_values%verbose = .true.
                                                i = i + 1
                                                case default
                                                print *, 'Unrecognized option:', trim(arg)
                                                end select
                        end do

                        ! check for arguments


                end subroutine

end module arg_parse


module f32_convert
        use iso_c_binding
        implicit none

        interface
                pure function float_to_half_c(x) bind(C, name="float_to_half")
                        use iso_c_binding
                        real(c_float), value :: x
                        integer(c_int16_t) :: float_to_half_c
                end function float_to_half_c

                pure function half_to_float_c(h) bind(C, name="half_to_float")
                        use iso_c_binding
                        integer(c_int16_t), value :: h
                        real(c_float) :: half_to_float_c
                end function half_to_float_c
        end interface

        real :: f32_lookup_table(-32768:32766)
        integer(2) :: v

        contains

                subroutine build_f32_lookup_table
                        do v = -32768,32766
                        f32_lookup_table(v) = half_to_float_c(v)
                        end do
        
                end subroutine

end module f32_convert

program llama2 
        use iso_c_binding
        use precision_module
        use weight_module
        use arg_parse
        use omp_lib
        use f32_convert, only: float_to_half_c, half_to_float_c, build_f32_lookup_table,&
                &f32_lookup_table

        implicit none


        
        ! weights and states
        INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        type(TransformerWeights) :: weights
        logical :: shared_weights
        integer :: head_size, tmp
        type(config) :: conf
        type(RunState) :: s
        real(kind=wp), allocatable :: logits(:)
        real(kind=wp), allocatable :: freq_buf(:)
        real(kind=wp), allocatable :: temp1(:), temp2(:,:), temp3(:,:,:)
        integer :: l
        ! anything that has n_layers, load in one layer at a time
        !, temp3(:,:,:)

        !for the tokens

        integer :: pos
        integer :: token        
        real(kind=wp) :: score
        integer :: tok_len, max_len, n
        !integer :: vocab_size = 32000
        character(:), allocatable :: tmpstr
        character(:), dimension(:), allocatable :: vocab
        real(kind=wp),allocatable :: scores(:)
        integer, allocatable :: prompt_tokens(:)
        integer, allocatable :: vocab_len(:)
        
        ! command line arguments
        !integer :: num_args
        !character(64) :: arg
        type (args) :: arg_values
        real :: temperature
        character(:), allocatable :: prompt
        logical :: verbose

        ! timing
        real(kind=wp) :: t_ms_start, t_ms_end

        call parse_args(arg_values)

        verbose = arg_values%verbose

        ! open the model file 
        open(UNIT=5, FILE=arg_values%model_file, FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
                ! config
                read(5) emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
                
                if (verbose) then
                        print *, "Embedding dimension: ", emb_dim
                        print *, "Hidden dimension: ", hidden_dim
                        print *, "Layers: ", n_layers
                        print *, "Heads: ", n_heads
                        print *, "kv Heads: ", n_kv_heads
                        print *, "Vocabulary Size: ", vocab_size
                        print *, "Sequence Length: ", seq_len

                end if 

                if (vocab_size > 0) then 
                        shared_weights = .true.
                else
                        shared_weights = .false.
                        vocab_size = -vocab_size
                end if 

                ! once we know the config sizes, allocate the arrays
                ! allocate temp, read in, convert to half precision
                
                allocate(weights%token_embedding_table(emb_dim,vocab_size))
                allocate(temp2(emb_dim,vocab_size))
                read(5) temp2
                weights%token_embedding_table = v_float_to_half_c2(temp2)
                deallocate(temp2)
                
                if (verbose) then
                        print *, "loaded embedding weights:", size(weights%token_embedding_table)
                end if 

                allocate(weights%rms_att_weight(emb_dim,n_layers))
                !allocate(temp2(emb_dim,n_layers))
                read(5) weights%rms_att_weight
                !weights%rms_att_weight = v_float_to_half_c2(temp2)
                !deallocate(temp2)

                if (verbose) then
                        print *, "loaded rms att weights:", size(weights%rms_att_weight)
                end if


                allocate(weights%wq(emb_dim,emb_dim,n_layers))
                allocate(temp2(emb_dim,emb_dim))
                do l = 1,n_layers
                !allocate(temp2(emb_dim,emb_dim,n_layers))
                read(5) temp2
                weights%wq(:,:,l) = v_float_to_half_c2(temp2)
                end do
                !deallocate(temp2)

                if (verbose) then
                        print *, "loaded wq weights:", size(weights%wq)
                end if
                
                allocate(weights%wk(emb_dim,emb_dim,n_layers))
                !allocate(temp2(emb_dim,emb_dim))
                do l = 1,n_layers
                !allocate(temp2(emb_dim,emb_dim,n_layers))
                read(5) temp2
                weights%wk(:,:,l) = v_float_to_half_c2(temp2)
                end do
                !deallocate(temp2)


                if (verbose) then
                        print *, "loaded wk weights:", size(weights%wk)
                end if
                
                allocate(weights%wv(emb_dim,emb_dim,n_layers))
                !allocate(temp2(emb_dim,emb_dim))
                do l = 1,n_layers
                !allocate(temp2(emb_dim,emb_dim,n_layers))
                read(5) temp2
                weights%wv(:,:,l) = v_float_to_half_c2(temp2)
                end do
                !deallocate(temp2)

        

                if (verbose) then
                        print *, "loaded wv weights:", size(weights%wv)
                end if
                
                allocate(weights%wo(emb_dim,emb_dim,n_layers))
                !allocate(temp2(emb_dim,emb_dim))
                do l = 1,n_layers
                !allocate(temp2(emb_dim,emb_dim,n_layers))
                read(5) temp2
                weights%wo(:,:,l) = v_float_to_half_c2(temp2)
                end do
                deallocate(temp2)



                if (verbose) then
                        print *, "loaded wo weights:", size(weights%wo)
                end if

                allocate(weights%rms_ffn_weight(emb_dim,n_layers))
                !allocate(temp2(emb_dim,n_layers))
                read(5) weights%rms_ffn_weight
                !weights%rms_ffn_weight = v_float_to_half_c2(temp2)
                !deallocate(temp2)

                if (verbose) then
                        print *, "loaded rms ffn  weights:", size(weights%rms_ffn_weight)
                end if

                allocate(weights%w1(emb_dim,hidden_dim,n_layers))
                allocate(temp2(emb_dim,hidden_dim))
                do l = 1,n_layers
                read(5) temp2
                weights%w1(:,:,l) = v_float_to_half_c2(temp2)
                end do
                deallocate(temp2)

                if (verbose) then
                        print *, "loaded w1 weights:", size(weights%w1)
                end if

                allocate(weights%w2(hidden_dim,emb_dim,n_layers))
                allocate(temp2(hidden_dim,emb_dim))
                do l = 1,n_layers
                read(5) temp2
                weights%w2(:,:,l) = v_float_to_half_c2(temp2)
                end do
                deallocate(temp2)

                if (verbose) then
                        print *, "loaded w2 weights:", size(weights%w2)
                end if

                allocate(weights%w3(emb_dim,hidden_dim,n_layers))
                allocate(temp2(emb_dim,hidden_dim))
                do l = 1,n_layers
                read(5) temp2
                weights%w3(:,:,l) = v_float_to_half_c2(temp2)
                end do
                deallocate(temp2)

                if (verbose) then
                        print *, "loaded w3 weights:", size(weights%w3)
                end if

                allocate(weights%rms_final_weight(emb_dim))
                !allocate(temp1(emb_dim))
                read(5) weights%rms_final_weight
                !weights%rms_final_weight = v_float_to_half_c(temp1)
                !deallocate(temp1)

                if (verbose) then
                        print *, "loaded rms_final weights:", size(weights%rms_final_weight)
                end if

                head_size = emb_dim / n_heads

                allocate(weights%freq_cis_real(head_size/2,seq_len))
                !allocate(temp2(head_size/2,seq_len))
                read(5) weights%freq_cis_real
                !weights%freq_cis_real = v_float_to_half_c2(temp2)
                ! deallocate(temp2)

                if (verbose) then
                        print *, "loaded freq cis real  weights:", size(weights%freq_cis_real)
                end if
                
                allocate(weights%freq_cis_imag(head_size/2,seq_len))
                !allocate(temp2(head_size/2,seq_len))
                read(5) weights%freq_cis_imag
                !weights%freq_cis_imag = v_float_to_half_c2(temp2)
                !deallocate(temp2)

                if (verbose) then
                        print *, "loaded freq_cis_imag weights:", size(weights%freq_cis_imag)
                end if


                if (.not. shared_weights) then
                        allocate(weights%wcls(emb_dim,vocab_size))
                        allocate(temp2(emb_dim,vocab_size))
                        read(5) temp2
                        weights%wcls = v_float_to_half_c2(temp2)
                        deallocate(temp2)

                        if (verbose) then
                                print *, "loaded wcls weights:", size(weights%wcls)
                        end if

                end if

        close(5) 

        if (verbose) then
                print *, "Loaded weights"
        end if

        ! config
        conf%emb_dim = emb_dim
        conf%hidden_dim = hidden_dim
        conf%n_layers = n_layers
        conf%n_heads = n_heads
        conf%n_kv_heads = n_kv_heads
        conf%vocab_size = vocab_size
        conf%seq_len = seq_len

        ! state dict
        allocate(s%att(seq_len,n_heads))
        allocate(s%key_cache(emb_dim,seq_len,n_layers))
        allocate(s%value_cache(emb_dim,seq_len,n_layers))

        ! not needed
        s%att(:,:) = 0
        s%key_cache(:,:,:) = 0
        s%value_cache(:,:,:) = 0
        s%times = 0

        ! read in token vocab
        open(UNIT=5, FILE=arg_values%tokenizer, FORM="UNFORMATTED",&
               & ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")

                read(5) max_len

                ! in fortran, all strings have to be the same length
                allocate(character(len=max_len) ::  vocab(vocab_size))
                allocate(scores(vocab_size))
                allocate(vocab_len(vocab_size))


                do n = 1,vocab_size



                read(5) score
                read(5) tok_len
                allocate (character(tok_len) :: tmpstr)
                read(5) tmpstr

                vocab(n) = tmpstr
                scores(n) = score
                ! we track the length of each token to preserve trailing whitespace
                vocab_len(n) = tok_len

                deallocate(tmpstr)

                end do

        close(5)

        ! __main__ part

        ! build lookup table
        call build_f32_lookup_table

        temperature = arg_values%temperature
        prompt = arg_values%prompt

        if (arg_values%n <= seq_len) then
                seq_len = arg_values%n
        else
                print *, arg_values%n, "greater than maxinum squence length"
                print *, "set to", seq_len
        end if

        t_ms_start = 0

        ! encode the prompt
        prompt_tokens = bpe_encode(prompt)

        ! indexing starts at 1, s is <s> BOS token        
            
        token = 2
        
        ! autoregressive model. get the next token from the last
        do pos = 1,seq_len
                logits = transformer(token,pos,conf,s,weights)
              
                if (pos <= size(prompt_tokens)) then

                        token = prompt_tokens(pos)

                else
                        if (temperature == 0) then
                                token = maxloc(logits,DIM=1)
                        else
                                logits = softmax(logits/temperature,vocab_size)
                                token = sample(logits)
                        end if
                end if

                ! here we kept track of the length to display each token properly
                write (*,fmt="(A)", advance="no") vocab(token)(1:vocab_len(token))
                
                ! start after first token as in llama2.c
                if (t_ms_start == 0) then
                        t_ms_start = time_ms()
                end if
        end do
        t_ms_end = time_ms()
        print *,""
        print *, "Inference time: ", (t_ms_end-t_ms_start)/1000, " seconds" 
        print *, 1000*(seq_len-1)/(t_ms_end-t_ms_start), "tokens/second"
        print *, "Timings"
        do l = 1,5
                print *, l, s%times(l)/seq_len
        end do
        ! end of __main__       

! functions 
contains 

        ! group the matmuls from MLP layers + nonlinearity together
        function p_mlp(xb,row,l, w) result(hb)
                real(kind=wp) :: xb(:)
                type(TransformerWeights), intent(in) :: w
                real(kind=wp) :: temp(size(xb))        
                integer :: row, l
                real(kind=wp) :: hb, hb2
                temp = v_half_to_float_lookup(w%w1(:,row,l))
                hb = dot_product(xb,temp)
                temp = v_half_to_float_lookup(w%w3(:,row,l))
                hb2 = dot_product(xb,temp)

                hb = hb*(1/(1+exp(-hb)))

                hb = hb*hb2
        end function
       
        ! group up front projection matmuls
        function p_proj(xb,row,l,w) result(p)
                real(kind=wp) :: xb(:)
                type(TransformerWeights), intent(in) :: w
                real(kind=wp) :: temp(size(xb))
                integer :: row, l
                real(kind=wp) :: p(3)

                temp = v_half_to_float_lookup(w%wq(:,row,l))
                p(1) = dot_product(xb,temp)
                temp = v_half_to_float_lookup(w%wk(:,row,l))
                p(2) = dot_product(xb,temp)
                temp = v_half_to_float_lookup(w%wv(:,row,l))
                p(3) = dot_product(xb,temp)

        end function

        function vm_matmul(a,b) result(c)
                real(kind=wp) :: a(:)
                real(kind=wp) :: b(:,:)
                integer, allocatable :: s(:)
                real(kind=wp), allocatable :: c(:)
                integer :: i,j
                real(kind=wp) :: val

                s = shape(b)
                allocate(c(s(2)))
                c (:) = 0

                !$OMP PARALLEL DO
                do i=1,s(2)
                        val = 0.0
                        !!$OMP PARALLEL DO PRIVATE(j) REDUCTION(+:val)
                        do j=1,size(a)
                                val = val + a(j)*b(j,i)
                        end do
                        !!$OMP END PARALLEL DO
                        c(i) = val

                end do
                !$OMP END PARALLEL DO




        end function

        !with loops
        function v_half_to_float_c(h)
                integer(2), intent(in) :: h(:)
                real(kind=wp) :: v_half_to_float_c (size(h))
                integer :: i
                !$OMP PARALLEL DO PRIVATE(i)
                do i=1,size(h)
                v_half_to_float_c(i) = half_to_float_c(h(i))
                end do 
                !$OMP END PARALLEL DO
        end function
       
        function v_half_to_float_lookup(h)
                integer(2), intent(in) :: h(:)
                real(kind=wp) :: v_half_to_float_lookup (size(h))
                integer :: i
                !$OMP PARALLEL DO PRIVATE(i)
                do i=1,size(h)
                v_half_to_float_lookup(i) = f32_lookup_table(h(i))
                end do
                !$OMP END PARALLEL DO
        end function

        pure function v_float_to_half_c(r)
                real(kind=wp), intent(in) :: r(:)
                integer(2) :: v_float_to_half_c (size(r))
                integer :: i
                do i = 1,size(r)
                v_float_to_half_c(i) = float_to_half_c(r(i))
                end do
        end function

        function v_half_to_float_c2(h)
                integer(2), intent(in) :: h(:,:)
                real(kind=wp) :: v_half_to_float_c2(size(h,1), size(h,2))
                integer :: i,j
                !$OMP PARALLEL DO COLLAPSE (2)
                do j = 1,size(h,2)
                        do i = 1,size(h,1)
                            v_half_to_float_c2(i,j) = half_to_float_c(h(i,j))
                        end do
                end do 
                !$OMP END PARALLEL DO   
        end function

        function v_half_to_float_lookup2(h)
                integer(2), intent(in) :: h(:,:)
                real(kind=wp) :: v_half_to_float_lookup2(size(h,1), size(h,2))
                integer :: i,j
                !$OMP PARALLEL DO COLLAPSE (2)
                do j = 1,size(h,2)
                        do i = 1,size(h,1)
                            v_half_to_float_lookup2(i,j) = f32_lookup_table(h(i,j))
                        end do
                end do
                !$OMP END PARALLEL DO   
        end function



        function v_float_to_half_c2(r)
                real(kind=wp), intent(in) :: r(:,:)
                integer(2) :: v_float_to_half_c2(size(r,1), size(r,2))
                integer :: i,j
                !$OMP PARALLEL DO COLLAPSE (2)
                do j = 1,size(r,2)
                        do i = 1,size(r,1)
                            v_float_to_half_c2(i,j) = float_to_half_c(r(i,j))
                        end do
                end do
                !$OMP END PARALLEL DO
        end function

        pure function v_float_to_half_c3(r)
                real(kind=wp), intent(in) :: r(:,:,:)
                integer(2) :: v_float_to_half_c3(size(r,1), size(r,2), size(r,3))
                v_float_to_half_c3 = reshape(v_float_to_half_c(&
                        &reshape(r, [size(r)])), [size(r,1), size(r,2), size(r,3)])
        end function
        
        function time_ms() result(t_ms)
                real(kind=wp) :: t_ms
                integer(4) :: ms
                !call cpu_time(t_ms)
                call system_clock(ms)
                t_ms = real(ms)
        end function



        ! sample from softmax probabilities  
        function sample(p) result(i)
              real(kind=wp) :: p(:)
              integer :: i
              real(kind=wp) :: r, cdf

              call random_number(r)
              cdf = 0

              do i=1,size(p)
              cdf = cdf + p(i)
              if (r<cdf) then
                      return
              end if
              
              end do

              i = size(p)
      

        end function 
      
        ! normalize and apply weigths. Note fortran built in dot product    
        function rmsnorm(x,w) result(xr)
              real(kind=wp) :: x(:), w(:)
              real(kind=wp) :: xr(size(x))
              real(kind=wp) :: xn
              xn = sqrt(dot_product(x,x)/size(x)+1e-5)

              xr = x*w/xn
        end function

        function trmsnorm(x,w,xn) result(xr)
              real(kind=wp) :: x, w
              real(kind=wp) :: xr
              real(kind=wp) :: xn
              !xn = sqrt(dot_product(x,x)/size(x)+1e-5)

              xr = x*w/xn
        end function
      
        ! declared as "pure" for potentiall parallelization of heads
        pure function softmax(x,s) result (p)
              real(kind=wp), intent(in) :: x(:)
              integer, intent(in) :: s
              real(kind=wp) :: p(size(x))
              real(kind=wp) :: xi(s)

              p(:) = 0
              xi = exp(x(:s)-maxval(x(:s)))
              p(:s) = xi/sum(xi) 

        end function 

      ! where the action isa
      ! see https://jaykmody.com/blog/gpt-from-scratch/ and ref therin for how attention works
        function transformer(token, pos, p, s, w) result(logits)
                integer, intent(in) :: token, pos
                type(Config), intent(in) :: p
                type(Runstate) :: s
                type(TransformerWeights), intent(in) :: w
                real(kind=wp) :: logits(p%vocab_size)

                real(kind=wp) :: x(p%emb_dim)
                real(kind=wp) :: xb(p%emb_dim)
                real(kind=wp) :: freq_cis_real_row(p%emb_dim/p%n_heads/2)
                real(kind=wp) :: freq_cis_imag_row(p%emb_dim/p%n_heads/2)

                ! embeddings
                real(kind=wp) :: proj(3)
                real(kind=wp) :: q(emb_dim)
                real(kind=wp) :: k(emb_dim)
                real(kind=wp) :: v(emb_dim)
      
                ! position encoding  
                real(kind=wp) :: q0, q1, k0, k1, fcr, fci

                ! attention
                real(kind=wp) :: q_t(p%emb_dim/p%n_heads)
                real(kind=wp) :: k_t(p%emb_dim/p%n_heads)
                real(kind=wp) :: v_t(p%emb_dim/p%n_heads)
                real(kind=wp) :: xbh(p%emb_dim/p%n_heads)
                real(kind=wp) :: a

                ! fc layers
                real(kind=wp) :: hb(hidden_dim)
                real(kind=wp) :: hb2(hidden_dim)
      
                ! counters etc
                integer :: l, i, h, t, head_size, ix
                real(kind=wp) :: time


                head_size = p%emb_dim/p%n_heads

                logits(:) = 0

                ! convert precision        
                x = v_half_to_float_lookup(w%token_embedding_table(:,token))

                freq_cis_real_row = w%freq_cis_real(:,pos)
                freq_cis_imag_row = w%freq_cis_imag(:,pos)

                do l = 1,p%n_layers
                        
                        ! embed and project
                        time = time_ms()
                        xb = rmsnorm(x,w%rms_att_weight(:,l))
                        !$OMP PARALLEL DO PRIVATE(ix,proj)
                        do ix = 1,p%emb_dim
                        proj = p_proj(xb,ix,l,w)
                        q(ix) = proj(1)
                        k(ix) = proj(2)
                        v(ix) = proj(3)
                        end do
                        !$OMP END PARALLEL DO
                        s%times(1) = s%times(1) + (time_ms()-time)
                        ! position encoding
        
                        time = time_ms()
                        do h = 0,(p%n_heads-1)
                                do i = 1,head_size,2

                                q0 = q(h*head_size+i)  
                                q1 = q(h*head_size+i+1)
                                k0 = k(h*head_size+i)
                                k1 = k(h*head_size+i+1)
                                fcr = freq_cis_real_row((i-1)/2+1)
                                fci = freq_cis_imag_row((i-1)/2+1)
                                q(h*head_size+i) = q0 * fcr - q1 * fci
                                q(h*head_size+i+1) = q0 * fci + q1 * fcr
                                k(h*head_size+i) = k0 * fcr - k1 * fci
                                k(h*head_size+i+1) = k0 * fci + k1 * fcr
          
                                end do
                        end do
                        s%times(2) = s%times(2) + (time_ms()-time)

                        ! cache k and v for this position
                        s%key_cache(:,pos,l) = k
                        s%value_cache(:,pos,l) = v 

                        xb(:) = 0
                        
                        ! multi head attention and fc layers
                        time = time_ms()
                        !$OMP PARALLEL DO PRIVATE(h, q_t, k_t, xbh, t, v_t, a)
                        do h = 0,(p%n_heads-1)        
                        ! alternately uncomment below to make explicitly concurrent 
                        !do concurrent (h =1:(p%n_heads-1))

                        q_t = q((h*head_size+1):((h+1)*head_size))
          
                        do t = 1,(pos)
                        k_t = s%key_cache((h*head_size+1):((h+1)*head_size),t,l)
                        s%att(t,h+1) = dot_product(q_t,k_t)/sqrt(real(head_size,wp))
                        end do  
          
                        ! beginning to POS, inclusive. so if pos = 1, there is 1...      
                        s%att(:,h+1) = softmax(s%att(:,h+1),pos)
                        xbh(:) = 0  
                        
                        do t = 1,(pos) 
                        v_t = s%value_cache((h*head_size+1):((h+1)*head_size),t,l)      
                        a = s%att(t,h+1)
                        xbh = xbh + a*v_t  
                        end do     
        
                        xb((h*head_size+1):((h+1)*head_size)) = xbh
                       
                        end do  
                        !$OMP END PARALLEL DO
                        s%times(3) = s%times(3) + (time_ms() - time)

                        time = time_ms()
                        x = x + vm_matmul(xb, v_half_to_float_lookup2(w%wo(:,:,l)))

                        xb = rmsnorm(x,w%rms_ffn_weight(:,l))
          
                        !$OMP PARALLEL DO PRIVATE(ix)
                        do ix = 1,p%hidden_dim
                        hb(ix) = p_mlp(xb,ix,l,w) 
                        end do
                        !$OMP END PARALLEL DO
                        
                        xb = vm_matmul(hb,v_half_to_float_lookup2(w%w2(:,:,l)))
                        x = x + xb

                        s%times(4) = s%times(4) + (time_ms() - time)

                end do

                time = time_ms()
                x = rmsnorm(x, w%rms_final_weight)

      
      
                if (shared_weights) then
                        logits = vm_matmul(x,v_half_to_float_lookup2(w%token_embedding_table))
                else
                        logits = vm_matmul(x,v_half_to_float_lookup2(w%wcls))
                end if
                s%times(5) = s%times(5) + (time_ms() - time)

        end function

        ! lookup the encoding of a token
        function lookup(s,l) result(ind)
                character(len=*) :: s
                integer :: l
                integer :: i, ind

                do i = 1,size(vocab)
                if (vocab(i) == s .and. vocab_len(i)==l) then
                        ind = i
                        return
                end if
                end do
                ind = -1
        end function

        ! encode text into tokens
        function bpe_encode(text) result(tokens)

                character(len=*) :: text
                integer, allocatable :: tokens(:)
                integer, allocatable :: tmp_tokens(:)
                integer :: i, ind, best_id, t1, t2
                real(kind=wp) :: score, best_score
                character(:), dimension(:), allocatable :: running_merge
                integer, allocatable :: running_merge_len(:)

                allocate(tokens(len(text)))

                do i = 1,len(text)
                        tokens(i) = lookup(text(i:i), 1)
                end do


                do while(1==1)

                        allocate(character(len=2*max_len) :: running_merge(size(tokens)-1))
                        allocate(running_merge_len(size(tokens)-1))

                        do i=1,(size(tokens)-1)

                                ! don't use trim, slice of to vocab_len
                                ! need to keep track of the true length of everything
                                t1 = vocab_len(tokens(i))
                                t2 = vocab_len(tokens(i+1))
                                running_merge(i) = vocab(tokens(i))(1:t1)//vocab(tokens(i+1))(1:t2)
                                running_merge_len(i) = t1+t2
                        end do


                        best_id = -1
                        best_score = -1e10
                        do i = 1,(size(tokens)-1)
                                ind = lookup(running_merge(i), running_merge_len(i))
                                if (ind > 0) then
                                        score = scores(ind)
                                        if (score > best_score) then
                                                best_score = score
                                                best_id = i
                                        end if

                                end if
                        end do


                        if (best_id == -1) then
                                exit
                        end if
                        
                        allocate(tmp_tokens(size(tokens)-1))
                        tmp_tokens(1:(best_id-1)) = tokens(1:(best_id-1))
                        tmp_tokens(best_id) = lookup(running_merge(best_id),running_merge_len(best_id))
                        tmp_tokens((best_id+1):) = tokens((best_id+2):)
                        deallocate(tokens)
                        call move_alloc(tmp_tokens,tokens)


                        deallocate(running_merge)
                        deallocate(running_merge_len)

                end do


        end function



end program llama2

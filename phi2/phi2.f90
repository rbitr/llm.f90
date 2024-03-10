! llama2.f90


module arg_parse
        implicit none

        type args
                real :: temperature
                character(:), allocatable :: model_file
                character(:), allocatable :: prompt
                character(:), allocatable :: tokenizer
                logical :: verbose, ak
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
                        arg_values%tokenizer = ""
                        arg_values%ak = .false.
                
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
                                                case ('--ak')
                                                ! llama2.c file format
                                                arg_values%ak = .true.
                                                i = i + 1
                                                case default
                                                print *, 'Unrecognized option:', trim(arg)
                                                stop
                                                end select
                        end do



                end subroutine

end module arg_parse

module f32_convert
        use iso_c_binding
        implicit none

        interface
                subroutine c_half_to_float_array(in_array, out_array, si) bind(C, name="half_to_float_array_simd")
                use iso_c_binding
                        integer(c_int16_t), intent(in) :: in_array(*)
                        real(c_float), intent(out) :: out_array(*)
                        integer(c_int), value :: si
                end subroutine c_half_to_float_array
        
                pure function c_dot_half_to_float_array(input_fp32, input_fp16, si) &
                                &bind(C, name="dot_product_fp16_fp32_v2")
                use iso_c_binding
                        integer(c_int16_t), intent(in) :: input_fp16(*)
                        real(c_float), intent(in) :: input_fp32(*)
                        real(c_float) :: c_dot_half_to_float_array
                        integer(c_int), value :: si
                end function c_dot_half_to_float_array
        
        end interface

        !float dot_product_fp16_fp32(const uint16_t* input_fp16, const float* input_fp32, int size)
        
end module f32_convert

program llama2 
        use iso_c_binding
        use precision_module
        use weight_module
        use arg_parse
        use read_ggml, only: load_ggml
        use pretokenize, only: pre_tokenize, decode, init
        use f32_convert
        !use omp_lib

        implicit none


        
        ! weights and states
        integer :: dummy(7)
        integer :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        !integer, parameter :: emb_dim = 2048
        !integer, parameter :: hidden_dim = 5632
        !integer, parameter :: n_layers = 22
        !integer, parameter :: n_heads = 32
        !integer, parameter :: n_kv_heads = 4
        !integer, parameter :: vocab_size = 32000
        !integer :: seq_len = 2048

        type(TransformerWeights16) :: weights
        logical :: shared_weights
        integer :: head_size, kv_head_size, tmp
        type(config) :: conf, dummy_conf
        type(RunState) :: s
        real(kind=wp), allocatable :: logits(:)
        real(kind=wp), allocatable :: freq_buf(:)
        real(kind=wp), allocatable :: temp1(:), temp2(:,:), temp3(:,:,:)
        integer :: l

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


        call init
        call parse_args(arg_values)

        verbose = arg_values%verbose
        

        call load_ggml(arg_values%model_file, weights, conf, vocab, scores, vocab_len, verbose)
        
        emb_dim = conf%emb_dim
        hidden_dim = conf%hidden_dim 
        n_layers = conf%n_layers 
        n_heads = conf%n_heads 
        n_kv_heads = conf%n_kv_heads 
        vocab_size = conf%vocab_size 
        seq_len = conf%seq_len

        max_len = maxval(vocab_len)
        head_size = emb_dim / n_heads
        kv_head_size = n_kv_heads * head_size
        shared_weights =.false.




        ! open the model file 


        if (verbose) then
                print *, "Loaded weights"
        end if


        ! config
        !conf%emb_dim = emb_dim
        !conf%hidden_dim = hidden_dim
        !conf%n_layers = n_layers
        !conf%n_heads = n_heads
        !conf%n_kv_heads = n_kv_heads
        !conf%vocab_size = vocab_size
        !conf%seq_len = seq_len
        !conf%kv_head_size = kv_head_size

        ! state dict
        allocate(s%att(seq_len,n_heads))
        allocate(s%key_cache(kv_head_size,seq_len,n_layers))
        allocate(s%value_cache(kv_head_size,seq_len,n_layers))

        ! not needed
        s%att(:,:) = 0
        s%key_cache(:,:,:) = 0
        s%value_cache(:,:,:) = 0
        s%times = 0

        if (arg_values%tokenizer /= "") then
        ! read in token vocab
        open(UNIT=5, FILE=arg_values%tokenizer, FORM="UNFORMATTED",&
               & ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")

                read(5) max_len

                ! in fortran, all strings have to be the same length
                if (allocated(vocab)) then
                        deallocate(vocab)
                        deallocate(scores)
                        deallocate(vocab_len)
                end if
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
        end if

        ! main part

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
        prompt = trim(pre_tokenize(prompt))
        if (verbose) then
                print *, "Pre-tokenized prompt: ", prompt
        end if
        prompt_tokens = bpe_encode(prompt)
        
        if (verbose) then
        do pos=1,size(prompt_tokens)
        print *, prompt_tokens(pos), vocab(prompt_tokens(pos))
        end do
        end if 
        
        ! indexing starts at 1, s is <s> BOS token        
                   
        token = 50257
        
        ! autoregressive model. get the next token from the last
        do pos = 1,seq_len
                logits = transformer(token,pos,s,weights)
              
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
                write (*,fmt="(A)", advance="no") decode(vocab(token),vocab_len(token))!(1:vocab_len(token))
                
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
        
        function gelu(x) result(y)
                real(kind=wp), intent(in) :: x(:)
                real(kind=wp) :: y(size(x))
                y = 0.5 * x * (1 + tanh(sqrt(2 / 3.1415926536) * (x + 0.044715 * x**3)))
        end function
        
        function v_half_to_float_c(h)
                integer(2), intent(in) :: h(:)
                real(kind=wp) :: v_half_to_float_c (size(h))
                integer :: i
                call c_half_to_float_array(h,v_half_to_float_c,size(h))
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
        !function rmsnorm(x,w) result(xr)
        !      real(kind=wp), intent(in) :: x(emb_dim), w(emb_dim)
        !      real(kind=wp) :: xr(emb_dim)
        !      real(kind=wp) :: xn
        !      xn = sqrt(dot_product(x,x)/size(x)+1e-5)
        !
        !      xr = x*w/xn
        !end function

        function rmsnorm(x,w,b) result(xr)
              real(kind=wp), intent(in) :: x(emb_dim), w(emb_dim), b(emb_dim)
              real(kind=wp) :: xr(emb_dim)
              real(kind=wp) :: xn
              xn = sqrt(dot_product(x,x)/size(x)+1e-5) !9.999999747378752e-06
        
              xr = x*w/xn+b
        end function

        function layer_norm(x,w,b) result(xr)
              real(kind=wp) :: x(:), w(:), b(:)
              real(kind=wp) :: xr(size(x))
              real(kind=wp) :: xmean(size(x)), xvar(size(x))
              real(kind=wp) :: xn
              !print *, "A"
              xmean = sum(x)/size(x) !spread(sum(x,dim=1)/size(x,1),1,size(x,1))
              xvar = sum( (x-xmean)*(x-xmean)) / size(x)
              xr = (x - xmean) / sqrt(xvar + 1e-5)
              !print *, "B"
              xr = xr*w + b
        end function


        !function trmsnorm(x,w,xn) result(xr)
        !      real(kind=wp) :: x, w
        !      real(kind=wp) :: xr
        !      real(kind=wp) :: xn
        !      !xn = sqrt(dot_product(x,x)/size(x)+1e-5)
        !
        !      xr = x*w/xn
        !end function
      
        pure function softmax(x,s) result (p)
              real(kind=wp), intent(in) :: x(:)
              integer, intent(in) :: s
              real(kind=wp) :: p(size(x))
              real(kind=wp) :: xi(s)

              p(:) = 0
              xi = exp(x(:s)-maxval(x(:s)))
              p(:s) = xi/sum(xi) 

        end function 

         pure function softmax_sl(x,s) result (p)
              real(kind=wp), intent(in) :: x(seq_len)
              integer, intent(in) :: s
              real(kind=wp) :: p(seq_len)
              real(kind=wp) :: xi(s)

              p(:) = 0
              xi = exp(x(:s)-maxval(x(:s)))
              p(:s) = xi/sum(xi)

        end function

        function rotate_half(x) result(y)
                real(kind=wp), intent(in) :: x(:)
                real(kind=wp)  :: y(size(x))
                y(: size(x)/2) = -x( size(x)/2 + 1 :)
                y(size(x)/2 + 1 :) = x(:size(x)/2)


        end function


        function transformer(token, pos, s, w) result(logits)
                integer, intent(in) :: token, pos
                !type(Config), intent(in) :: p
                type(Runstate) :: s
                type(TransformerWeights16), intent(in) :: w
                real(kind=wp) :: logits(vocab_size)

                integer, parameter :: rope_dim = 32
                real(kind=wp) :: x(emb_dim)
                real(kind=wp) :: xb(emb_dim)
                real(kind=wp) :: xt(emb_dim)

                ! embeddings
                real(kind=wp), target :: qkv(emb_dim+2*kv_head_size)
                real(kind=wp), pointer :: q(:), k(:), v(:)
      
                ! position encoding  
                real(kind=wp) :: q0, q1, k0, k1, fcr, fci, v0, v1, freq, rval
                integer :: head_dim

                ! attention
                real(kind=wp) :: q_t(emb_dim/n_heads)
                real(kind=wp) :: k_t(emb_dim/n_heads)
                real(kind=wp) :: v_t(emb_dim/n_heads)
                real(kind=wp) :: xbh(emb_dim/n_heads)
                real(kind=wp) :: a
                integer :: kv_mul

                ! fc layers
                real(kind=wp), target :: hb(hidden_dim), h0(emb_dim)

                ! counters etc
                integer :: l, i,j, h, t, head_size, ix
                real(kind=wp) :: time
                real :: steps(16) = (/(i, i=0,15)/)
                real :: freqs(rope_dim), cosf(rope_dim), sinf(rope_dim)!, qq(rope_dim), kk(rope_dim)
                !integer :: rope_dim
                

                head_size = emb_dim/n_heads
                q => qkv(1:emb_dim)
                k => qkv((emb_dim+1):(emb_dim+kv_head_size))
                v => qkv((emb_dim+kv_head_size+1):(emb_dim+2*kv_head_size))
                
                do i=1,rope_dim,2
                                freqs(int((i-1)/2)+1) = 1.0 / (10000.0 ** (real(i-1,kind=wp) / real(rope_dim,kind=wp)))
                end do

                freqs(rope_dim/2+1:) = freqs(1:rope_dim/2)

                cosf = cos(freqs*(pos-1))
                sinf = sin(freqs*(pos-1))

                
                

                ! convert precision        
                !x = fp16_to_real32(w%token_embedding_table(:,token))
                x = v_half_to_float_c(w%token_embedding_table(:,token))


                do l = 1,n_layers
                        
                        ! embed and project
                        time = time_ms()
                        xb = layer_norm(x,w%rms_att_weight(:,l),w%rms_att_bias(:,l))
                        h0 = xb
                        
                        !do ix = 1,size(qkv)
                        do ix=1,size(qkv)
                        !qkv(ix) = dot_product(xb,v_half_to_float_c(w%wqkv(:,ix,l)))
                        qkv(ix) = c_dot_half_to_float_array(xb,w%wqkv(:,ix,l),emb_dim)
                        end do
                        qkv = qkv + w%bqkv(:,l)
                        
                        
                        s%times(1) = s%times(1) + (time_ms()-time)
                        ! position encoding
        
                        time = time_ms()
                        ! check that this doens't add any time
                        
                        do h=1,n_heads
                        associate (qq => q((h-1)*head_size+1 : (h-1)*head_size+rope_dim), &
                                        &kk => k((h-1)*head_size+1 : (h-1)*head_size+rope_dim))
                        qq = qq * cosf + rotate_half(qq) * sinf
                        kk = kk * cosf + rotate_half(kk) * sinf
                        
                        end associate 
                        end do
                        
                        
                        s%times(2) = s%times(2) + (time_ms()-time)

                        ! cache k and v for this position
                        s%key_cache(:,pos,l) = k
                        s%value_cache(:,pos,l) = v 

                        xb(:) = 0
                        
                        ! multi head attention and fc layers
                        time = time_ms()
                        
                        kv_mul = n_heads / n_kv_heads
                        
                        do h = 0,(n_heads-1)        

                        q_t = q((h*head_size+1):((h+1)*head_size))
          
                        do t = 1,(pos)
                        !k_t = s%key_cache((h*head_size+1):((h+1)*head_size),t,l)
                        ! for shared heads 
                        k_t = s%key_cache(((h/kv_mul)*head_size+1):(((h+1)/kv_mul)*head_size),t,l)
                        s%att(t,h+1) = dot_product(q_t/sqrt(real(head_size,wp)),k_t)                        
                        !s%att(t,h+1) = dot_product(q_t,k_t)/sqrt(real(head_size,wp))
                        end do  
          
                        ! beginning to POS, inclusive. so if pos = 1, there is 1...      
                        s%att(:,h+1) = softmax_sl(s%att(:,h+1),pos)
                        xbh(:) = 0  
                        
                        do t = 1,(pos) 
                        !v_t = s%value_cache((h*head_size+1):((h+1)*head_size),t,l)      
                        v_t = s%value_cache(((h/kv_mul)*head_size+1):(((h+1)/kv_mul)*head_size),t,l) 
                        a = s%att(t,h+1)
                        xbh = xbh + a*v_t  
                        end do     
        
                        xb((h*head_size+1):((h+1)*head_size)) = xbh
                       
                        end do  
                        s%times(3) = s%times(3) + (time_ms() - time)

                        time = time_ms()
                        
                        do ix=1,emb_dim
                        !x(ix) = x(ix) + dot_product(xb,v_half_to_float_c(w%wo(:,ix,l)))
                        xt(ix) = c_dot_half_to_float_array(xb,w%wo(:,ix,l),emb_dim)
                        end do
                        xt = xt + w%bo(:,l)
                        x = x + xt
                        
                        xb = h0 
                        !xb = rmsnorm(x,w%rms_ffn_weight(:,l))
          
                        do ix = 1,hidden_dim
                        !hb13(ix) = dot_product(xb,v_half_to_float_c(w%w13(:,ix,l)))
                        hb(ix) = c_dot_half_to_float_array(xb,w%wup(:,ix,l),emb_dim)
                        end do
                        hb = hb + w%bup(:,l)

                        !nonlinearity:
                        hb = gelu(hb)

                        do ix = 1,emb_dim
                        !x(ix) = x(ix) + dot_product(hb,v_half_to_float_c(w%w2(:,ix,l)))
                        xt(ix) = c_dot_half_to_float_array(hb,w%wdown(:,ix,l),hidden_dim)
                        end do
                        xt = xt + w%bdown(:,l)
                        x = x + xt
                        
                        s%times(4) = s%times(4) + (time_ms() - time)

                end do

                time = time_ms()
                x = layer_norm(x, w%rms_final_weight,w%rms_final_bias)

      
                !if (shared_weights) then
                !        logits = vm_matmul(x,(w%token_embedding_table))
                !else
                        do ix = 1,vocab_size
                                !logits(ix) = dot_product(x,v_half_to_float_c(w%wcls(:,ix)))
                                logits(ix) = c_dot_half_to_float_array(x,w%wcls(:,ix),emb_dim)
                        end do
                        logits = logits + w%bcls
                !end if
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
                integer :: i,j, ind, best_id, t1, t2
                real(kind=wp) :: score, best_score
                character(:), dimension(:), allocatable :: running_merge
                integer, allocatable :: running_merge_len(:)
                integer(1) :: t

                allocate(tmp_tokens(len(text)))

                i = 1
                j = 1
                
                do while (i <= len(text)) !i = 1,len(text)
                        !tokens(i) = lookup(text(i:i), 1)
                        ! should be able to either have 1 or 2 byte encoding
                        ! because all unicode has been byte encoded
                        read(text(i:i), "(A)") t
                        if (t >= 0 .and. t < 128) then
                        tmp_tokens(j) = lookup(text(i:i),1)
                        i = i + 1
                        else
                        tmp_tokens(j) = lookup(text(i:i+1),2)
                        i = i + 2
                        end if
                        j = j + 1
                        !if (verbose) then
                        !        print *, tmp_tokens(j-1)
                        !end if 
                end do

                allocate(tokens(j-1))
                do i = 1,(j-1)
                        tokens(i) = tmp_tokens(i)
                end do
                deallocate(tmp_tokens)


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

        pure elemental function fp16_to_real32(fp16_value) result(vx)
                integer(2), intent(in) :: fp16_value
                real(kind=wp) :: vx
                integer(4) :: int_value, si, ex, mantissa

                si = ibits(fp16_value, 15, 1)               ! Extract sign bit
                ex = ibits(fp16_value, 10, 5)           ! Extract exponent
                mantissa = ibits(fp16_value, 0, 10)           ! Extract mantissa

                ! Adjust exponent from FP16 bias (15) to FP32 bias (127)
                ex = ex - 15 + 127

        
                ! If exponent is all 1's, it's a special value (Inf or NaN)
                if (ex >= 142) then
                        ! Set exponent to all 1's for FP32
                        ex = 255
                        ! If mantissa is not zero, it's NaN
                        if (mantissa /= 0) then
                        ! Set mantissa to a non-zero value
                        mantissa = 1
                        endif
                elseif (ex <= 112) then
                        ! If the FP16 was denormalized, or too small for normalized FP32
                        ex = 0
                        mantissa = 0
                        else
                        ! Normalize the mantissa by adding the implicit leading 1 and adjusting exponent
                        mantissa = mantissa * 2**13
                endif


                ! Combine sign, adjusted exponent and mantissa
                int_value = (ishft(si, 31)) + (ishft(ex, 23)) + mantissa

                ! Transfer the bits back to real32
                vx = transfer(int_value, vx)

        end function fp16_to_real32

end program llama2

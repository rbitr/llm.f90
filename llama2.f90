! llama2.f90
! implementation of a toy llama model

! structs for reading weights, config information and state 
module weight_module
        implicit none
        type TransformerWeights
                real, allocatable :: token_embedding_table(:,:)
                real, allocatable :: rms_att_weight(:,:)
                real, allocatable :: rms_ffn_weight(:,:)
                real, allocatable :: wq(:,:,:)
                real, allocatable :: wk(:,:,:)
                real, allocatable :: wv(:,:,:)
                real, allocatable :: wo(:,:,:)
                real, allocatable :: w1(:,:,:)
                real, allocatable :: w2(:,:,:)
                real, allocatable :: w3(:,:,:)
                real, allocatable :: rms_final_weight(:)
                real, allocatable :: freq_cis_real(:,:)
                real, allocatable :: freq_cis_imag(:,:)
  
        end type TransformerWeights

        type Config
                INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        end type Config

        type RunState

                real, allocatable :: att(:,:)
                real, allocatable :: key_cache(:,:,:)
                real, allocatable :: value_cache(:,:,:)

        end type RunState

end module weight_module

program llama2 
        use weight_module
        
        ! weights and states
        INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        type(TransformerWeights) :: weights
        integer :: head_size, tmp
        type(config) :: conf
        type(RunState) :: s
        real, allocatable :: logits(:)
        real, allocatable :: freq_buf(:)

        !for the tokens

        integer :: pos
        integer :: token        
        real :: score
        integer :: tok_len, max_len, n
        !integer :: vocab_size = 32000
        character(:), allocatable :: tmpstr
        character(:), dimension(:), allocatable :: vocab
        real,allocatable :: scores(:)
        integer, allocatable :: prompt_tokens(:)
        integer, allocatable :: vocab_len(:)
        
        ! command line arguments
        integer :: num_args
        character(64) :: arg
        real :: temperature
        character(:), allocatable :: prompt

        ! open the model file 
        open(UNIT=5, FILE="stories15M.bin", FORM="UNFORMATTED", ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
                ! config
                read(5) emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len

                ! once we know the config sizes, allocate the arrays
                allocate(weights%token_embedding_table(emb_dim,vocab_size))
                allocate(weights%rms_att_weight(emb_dim,n_layers))

                allocate(weights%wq(emb_dim,emb_dim,n_layers))
                allocate(weights%wk(emb_dim,emb_dim,n_layers))
                allocate(weights%wv(emb_dim,emb_dim,n_layers))
                allocate(weights%wo(emb_dim,emb_dim,n_layers))

                allocate(weights%rms_ffn_weight(emb_dim,n_layers))

                allocate(weights%w1(emb_dim,hidden_dim,n_layers))
                allocate(weights%w2(hidden_dim,emb_dim,n_layers))
                allocate(weights%w3(emb_dim,hidden_dim,n_layers))

                allocate(weights%rms_final_weight(emb_dim))

                head_size = emb_dim / n_heads

                allocate(weights%freq_cis_real(head_size/2,seq_len))
                allocate(weights%freq_cis_imag(head_size/2,seq_len))

                ! read everything in
                read(5) weights%token_embedding_table
                read(5) weights%rms_att_weight
                read(5) weights%wq
                read(5) weights%wk
                read(5) weights%wv
                read(5) weights%wo
                read(5) weights%rms_ffn_weight
                read(5) weights%w1
                read(5) weights%w2
                read(5) weights%w3
                read(5) weights%rms_final_weight
                read(5) weights%freq_cis_real
                read(5) weights%freq_cis_imag

        close(5) 

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

        ! read in token vocab
        open(UNIT=5, FILE="tokenizer.bin", FORM="UNFORMATTED", ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")

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
        ! argparse
        num_args = command_argument_count()

        !print *, num_args

        if (num_args>0) then
                
                call get_command_argument(1,arg)
                read(arg,*) temperature
                !print *, "Temp: ", temperature

        else

                temperature = .9

        end if

        if (num_args>1) then
                call get_command_argument(2,arg)
                prompt = trim(arg)
                !print *, "Prompt: ", prompt 
        else

        prompt = ""

        end if

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

        end do
        print *,""
        ! end of __main__       

! functions 
contains 

        ! sample from softmax probabilities  
        function sample(p) result(i)
              real :: p(:)
              integer :: i
              real :: r, cdf

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
              real :: x(:), w(:)
              real :: xr(size(x))
              real :: xn
              xn = sqrt(dot_product(x,x)/size(x)+1e-5)

              xr = x*w/xn
        end function
      

        function softmax(x,s) result (p)
              real :: x(:)
              integer :: s
              real :: p(size(x))
              real :: xi(s)

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
                real :: logits(p%vocab_size)

                real :: x(p%emb_dim)
                real :: xb(p%emb_dim)
                real :: freq_cis_real_row(p%emb_dim/p%n_heads/2)
                real :: freq_cis_imag_row(p%emb_dim/p%n_heads/2)

                ! embeddings
                real :: q(emb_dim)
                real :: k(emb_dim)
                real :: v(emb_dim)
      
                ! position encoding  
                real :: q0, q1, k0, k1, fcr, fci

                ! attention
                real :: q_t(p%emb_dim/p%n_heads)
                real :: k_t(p%emb_dim/p%n_heads)
                real :: v_t(p%emb_dim/p%n_heads)
                real :: xbh(p%emb_dim/p%n_heads)
                real :: a

                ! fc layers
                real :: hb(hidden_dim)
                real :: hb2(hidden_dim)
      
                ! counters etc
                integer :: l, i, h, t, head_size


                head_size = p%emb_dim/p%n_heads

                logits(:) = 0

                x = w%token_embedding_table(:,token)

                freq_cis_real_row = w%freq_cis_real(:,pos)
                freq_cis_imag_row = w%freq_cis_imag(:,pos)

                do l = 1,p%n_layers
                        
                        ! embed and project
                        xb = rmsnorm(x,w%rms_att_weight(:,l)) 
        
                        q = matmul(xb,w%wq(:,:,l))
                        k = matmul(xb,w%wk(:,:,l))
                        v = matmul(xb,w%wv(:,:,l))
       
                        ! position encoding
        
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

                        ! cache k and v for this position
                        s%key_cache(:,pos,l) = k
                        s%value_cache(:,pos,l) = v 

                        xb(:) = 0
                        
                        ! multi head attention and fc layers
                        do h = 0,(p%n_heads-1)        
            
                        q_t = q((h*head_size+1):((h+1)*head_size))
          
                        do t = 1,(pos)
                        k_t = s%key_cache((h*head_size+1):((h+1)*head_size),t,l)
                        s%att(t,h+1) = dot_product(q_t,k_t)/sqrt(real(head_size))
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


                        x = x + matmul(xb, w%wo(:,:,l))

                        xb = rmsnorm(x,w%rms_ffn_weight(:,l))
          

                        hb = matmul(xb,w%w1(:,:,l))
                        hb2 = matmul(xb,w%w3(:,:,l))

                        hb = hb*(1/(1+exp(-hb)))

                        hb = hb*hb2
                        xb = matmul(hb,w%w2(:,:,l))

                        x = x + xb

                end do

                x = rmsnorm(x, w%rms_final_weight)

      
      

                logits = matmul(x,w%token_embedding_table)


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
                real :: score, best_score
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

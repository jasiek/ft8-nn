! jtdx_ft8_file - decode FT8 from WAV files with JTDX's own decoder.
!
! JTDX's decoder binary (jtdxjt9) cannot read files: JTDX deleted WSJT-X's
! file-reading branch from lib/jt9.f90 ("Decoder for JT9. Can run as the back
! end of JTDX only, with data placed in a shared memory region"), leaving only
! the jt9a() shared-memory path its GUI drives.  The decoder subroutines are
! untouched, so this driver re-adds file mode: it mirrors lib/jt9a.f90's
! ndiskdat branch (audio into dd8, nzhsym = 49 half-symbol blocks, which is
! also what live JTDX uses for FT8 - mainwindow.cpp m_hsymStop) and fills
! params_block the way MainWindow::decode() does, then calls JTDX's
! multimode_decoder.  Decode lines are printed by JTDX itself, from
! decoder.f90's ft8_decoded callback, in jt9's FT8 line format.
!
! Files are decoded in one process in the order given, so JTDX keeps the
! cross-slot state it has when running live: its even/odd message memory and
! the callsign hash table that resolves <...> calls.
!
!   jtdx_ft8_file [-a NFA] [-b NFB] [-d DEPTH] [-c CYCLES] [-s SENS] [-w] [-u] FILE...
!     -a/-b  decoder search range in Hz             (default 150 3100)
!     -d     ndepth 1..3                            (default 3 = deepest, JTDX default)
!     -c     nft8cycles 1..3 -> 3/6/9 passes        (default 1, JTDX default)
!     -s     FT8 sensitivity 0|1|2                  (default 1, JTDX default)
!              0 = minimum, 1 = lft8lowth (low sync thresholds),
!              2 = lft8lowth + lft8subpass
!     -w     SWL mode (nswl: 51 blocks, lower sync threshold, nft8swlcycles)
!     -u     keep same-slot duplicate messages (JTDX's HideFT8Dupes defaults
!            to true, and without it JTDX emits weak spurious copies of
!            messages it has just decoded)
!     -t     nmt, decoder threads over the band     (default 1)
program jtdx_ft8_file
  use, intrinsic :: iso_c_binding
  use prog_args
  use ft8_mod1, only : dd8
  include 'jt9com.f90'

  type(params_block) :: params
  character(len=1024) :: arg, fname
  character(len=16) :: base
  integer :: i, iarg, nargs, nfa, nfb, ndepth, ncycles, nsens, nthreads
  integer :: nutc, nblocks, nlastsam, nsamp, ios, ihh, imm, iss, k
  logical :: swl, firstfile, keepdupes
  integer(c_short), allocatable :: id2(:)

  nfa = 150; nfb = 3100; ndepth = 3; ncycles = 1; nsens = 1
  swl = .false.; nthreads = 1; keepdupes = .false.
  nargs = command_argument_count()

  ! ---- options -------------------------------------------------------------
  iarg = 1
  do while (iarg <= nargs)
     call get_command_argument(iarg, arg)
     if (arg(1:1) /= '-') exit
     select case (trim(arg))
     case ('-a'); iarg=iarg+1; call get_command_argument(iarg,arg); read(arg,*) nfa
     case ('-b'); iarg=iarg+1; call get_command_argument(iarg,arg); read(arg,*) nfb
     case ('-d'); iarg=iarg+1; call get_command_argument(iarg,arg); read(arg,*) ndepth
     case ('-c'); iarg=iarg+1; call get_command_argument(iarg,arg); read(arg,*) ncycles
     case ('-s'); iarg=iarg+1; call get_command_argument(iarg,arg); read(arg,*) nsens
     case ('-t'); iarg=iarg+1; call get_command_argument(iarg,arg); read(arg,*) nthreads
     case ('-w'); swl = .true.
     case ('-u'); keepdupes = .true.
     case default
        write(0,*) 'jtdx_ft8_file: unknown option ', trim(arg); stop 2
     end select
     iarg = iarg + 1
  end do
  if (iarg > nargs) then
     write(0,*) 'usage: jtdx_ft8_file [-a NFA] [-b NFB] [-d DEPTH] [-c CYCLES]' // &
                ' [-s SENS] [-w] [-u] [-t THREADS] FILE...'
     stop 2
  end if

  ! ---- params: MainWindow::decode() for FT8, reading from disk -------------
  params%mycall = ' '; params%mybcall = ' '
  params%hiscall = ' '; params%hisbcall = ' '; params%hisgrid = ' '
  params%listutc = 0
  params%napwid = 5                 ! HF (freqNominal < 30 MHz)
  params%nQSOProgress = 0
  params%nftx = 1500
  params%nutc = 0
  params%ntrperiod = 15
  params%nfqso = 1500
  params%npts8 = 0
  params%nfa = nfa
  params%nfsplit = 2700
  params%nfb = nfb
  params%ntol = 50                  ! "this value is not being used" for FT8
  params%kin = 0
  params%ndepth = ndepth
  params%ncandthin = 100            ! GUI default: no candidate thinning
  params%ndtcenter = 0
  params%nft8cycles = ncycles
  params%nft8swlcycles = ncycles
  params%ntxmode = 0
  params%nmode = 8                  ! FT8
  params%nlist = 0
  params%nranera = 3                ! Configuration.cpp defaults from here down
  params%ntrials10 = 1
  params%ntrialsrxf10 = 1
  params%naggressive = 1
  params%nharmonicsdepth = 0
  params%ntopfreq65 = 2700
  params%nprepass = 4
  params%nsdecatt = 1
  params%nlasttx = 0
  params%ndelay = 0
  params%nmt = nthreads
  params%nft8rxfsens = 2
  params%nft4depth = 1
  params%nsecbandchanged = 0

  params%ndiskdat = .true.          ! audio comes from a file
  params%newdat = .true.
  params%nagain = .false.
  params%nagainfil = .false.
  params%nswl = swl
  params%nfilter = .false.          ! no RX-frequency filter: decode the band
  params%nstophint = .false.
  params%nagcc = .false.
  params%nhint = .false.
  params%fmaskact = .false.
  params%showharmonics = .false.
  params%lft8lowth = (nsens >= 1)
  params%lft8subpass = (nsens >= 2)
  params%ltxing = .false.
  params%lhidetest = .false.        ! contest/telemetry hiding: GUI default off
  params%lhidetelemetry = .false.
  params%lhideft8dupes = .not. keepdupes   ! GUI default: hide them
  params%lhound = .false.
  params%lhidehash = .false.
  params%lcommonft8b = .true.       ! recordings are on the standard FT8 frequencies
  params%lmycallstd = .false.
  params%lhiscallstd = .false.
  params%lapmyc = .false.
  params%lmodechanged = .true.      ! first slot only, cleared below
  params%lbandchanged = .true.
  params%lenabledxcsearch = .false.
  params%lwidedxcsearch = .false.
  params%lmultinst = .false.
  params%lskiptx1 = .false.
  params%lforcesync = .false.
  params%learlystart = .false.

  ! nblocks: jt9a.f90's values for nmode 8 under ndiskdat
  if (swl) then; nblocks = 51
  else; nblocks = 49
  end if
  params%nzhsym = nblocks
  nlastsam = nblocks*3456

  allocate(id2(180000))
  firstfile = .true.

  do i = iarg, nargs
     call get_command_argument(i, fname)
     call read_wav(trim(fname), id2, nsamp, ios)
     if (ios /= 0) then
        write(0,'(a,i0,a,a)') 'jtdx_ft8_file: cannot read (ios=', ios, ') ', trim(fname)
        cycle
     end if

     ! nutc = HHMMSS from a YYMMDD_HHMMSS.wav name.  decoder.f90 needs
     ! mod(nutc,100) in {0,15,30,45}, else it drops its even/odd slot state.
     k = index(fname, '/', back=.true.)
     base = fname(k+1:k+13)
     read(base(8:9),  *, iostat=ios) ihh
     read(base(10:11),*, iostat=ios) imm
     read(base(12:13),*, iostat=ios) iss
     if (ios /= 0) then
        write(0,*) 'jtdx_ft8_file: bad slot name ', trim(fname); cycle
     end if
     nutc = 10000*ihh + 100*imm + iss
     params%nutc = nutc

     dd8 = 0.
     dd8(1:min(nlastsam,nsamp)) = id2(1:min(nlastsam,nsamp))

     write(*,'(a,a)') '#F ', trim(fname)
     call flush(6)
     call multimode_decoder(params)
     call flush(6)

     if (firstfile) then
        params%lmodechanged = .false.
        params%lbandchanged = .false.
        firstfile = .false.
     end if
  end do

contains

  ! Minimal RIFF reader: mono 16-bit PCM at 12 kHz, as collect/kiwi.py writes.
  subroutine read_wav(path, buf, nsamp, ios)
    character(len=*), intent(in) :: path
    integer(c_short), intent(out) :: buf(:)
    integer, intent(out) :: nsamp, ios
    character(len=4) :: tag
    integer :: lu, pos, csize, srate, n
    integer(c_short) :: nch, bits
    nsamp = 0
    open(newunit=lu, file=path, access='stream', form='unformatted', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) return
    read(lu, pos=1, iostat=ios) tag
    if (ios /= 0 .or. tag /= 'RIFF') then; ios = 1; close(lu); return; end if
    read(lu, pos=9, iostat=ios) tag
    if (ios /= 0 .or. tag /= 'WAVE') then; ios = 1; close(lu); return; end if
    pos = 13
    do
       read(lu, pos=pos, iostat=ios) tag
       if (ios /= 0) exit
       read(lu, pos=pos+4, iostat=ios) csize
       if (ios /= 0) exit
       if (tag == 'fmt ') then
          read(lu, pos=pos+10, iostat=ios) nch
          read(lu, pos=pos+12, iostat=ios) srate
          read(lu, pos=pos+22, iostat=ios) bits
          if (nch /= 1 .or. bits /= 16 .or. srate /= 12000) then
             write(0,*) 'jtdx_ft8_file: expected mono/16-bit/12 kHz, got ', &
                  nch, bits, srate
             ios = 1; close(lu); return
          end if
       else if (tag == 'data') then
          n = min(csize/2, size(buf))
          read(lu, pos=pos+8, iostat=ios) buf(1:n)
          if (ios /= 0) then; close(lu); return; end if
          nsamp = n
          close(lu)
          return
       end if
       pos = pos + 8 + csize + mod(csize, 2)
    end do
    ios = 1
    close(lu)
  end subroutine read_wav

end program jtdx_ft8_file

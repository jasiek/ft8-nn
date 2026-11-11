/* ft8nn_sidecar: libft8nn (this project's FT8 decoder) as WSJT-X's external FT8 decoder.
 *
 * WSJT-X 3.0 can hand its FT8 decoding to a long-running external process - a "sidecar" - named by
 * the environment variable WSJTX_FT8_DECODER.  The protocol (doc/ft8-sidecar-protocol.md in the
 * WSJT-X tree, Decoder/Ft8Sidecar.cpp its implementation) is one compact JSON object per line:
 *
 *   WSJT-X -> us, on stdin
 *     {"type":"hello","v":1,...}                             once, at startup
 *     {"type":"decode","v":1,"seq":N,...,"samples":172800}    one per decode pass, the JSON line
 *                                                            followed at once by samples*2 bytes of
 *                                                            mono s16le 12 kHz audio, sample 0 at
 *                                                            the start of the T/R period
 *     {"type":"quit","v":1}                                  at shutdown; exit within 1 s
 *
 *   us -> WSJT-X, on stdout (flushed line by line)
 *     {"seq":N,"snr":-12,"dt":0.3,"freq":1234,"msg":"CQ K1ABC FN42"[,"ap":1][,"lowconf":true]}
 *     {"seq":N,"done":true}                                  exactly one per request, always
 *
 *   Decodes go out as the decoder finalises them, a batch per subtraction pass, not all at the end:
 *   WSJT-X shows each line as it arrives (and auto-sequencing acts on it), and the first pass finds
 *   most of a slot's decodes in a fraction of the total time.
 *
 *   stderr is free-form text; WSJT-X copies it into its log.
 *
 * WSJT-X formats our decodes into jt9's FT8 output lines itself and feeds them through the same
 * code as jt9's output, so false-decode filtering, ALL.TXT, the Band Activity and Rx Frequency
 * windows, UDP, PSK Reporter and auto-sequencing behave as they do with jt9.  It only ever sends
 * FT8 here - every other mode, and Fox and SuperFox FT8, stay with its own jt9 - so there is
 * nothing to forward and no other mode to implement.
 *
 * With .wav files on the command line it decodes those instead and prints jt9's FT8 lines, which is
 * how the tests compare this path against a plain decode of the same audio.
 *
 * Usage: see --help; wsjtx/README.md has the install steps and the limits.
 */
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <time.h>
#include <unistd.h>

#include "json.h"
#include "linereader.h"
#include "wav.h"

extern "C" {
#include "../c/include/ft8nn.h"
}

static int const SAMPLE_RATE = 12000;
static int const SLOT_SAMPLES = 15 * SAMPLE_RATE;        // a full FT8 T/R period

static bool verbose = false;

static void say (char const * fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  fputs ("[ft8nn] ", stderr);
  vfprintf (stderr, fmt, ap);
  fputc ('\n', stderr);
  va_end (ap);
  fflush (stderr);
}

static void vsay (char const * fmt, ...)
{
  if (!verbose) return;
  va_list ap;
  va_start (ap, fmt);
  fputs ("[ft8nn] ", stderr);
  vfprintf (stderr, fmt, ap);
  fputc ('\n', stderr);
  va_end (ap);
  fflush (stderr);
}

static double monotonic ()
{
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ---------------------------------------------------------------- options */

struct Options
{
  std::string model;                    // --model, else $FT8NN_MODEL, else searched (find_model)
  int threads = 1;                      // --threads: ONNX Runtime intra-op threads
  int max_decodes = 200;                // --max-decodes per pass
  bool early = false;                   // --early: decode WSJT-X's early passes too
  bool range = true;                    // --no-range: keep decodes outside [nfa, nfb]
  bool help = false;
  std::vector<std::string> files;       // file mode: decode these instead of talking to WSJT-X
};

static char const * const usage_text =
  "ft8nn_sidecar: this project's FT8 decoder as WSJT-X's external FT8 decoder.\n"
  "\n"
  "  ft8nn_sidecar [options]                  speak the sidecar protocol on stdin/stdout\n"
  "  ft8nn_sidecar [options] FILE.wav...      decode 15 s WAV files and print jt9 lines\n"
  "\n"
  "Options:\n"
  "  --model PATH      ft8nn.onnx (default: $FT8NN_MODEL, then ft8nn.onnx next to this\n"
  "                    binary, in $WSJTX_DATA_DIR / $WSJTX_SHARE_DIR, or ../models/v3)\n"
  "  --threads N       ONNX Runtime threads for the neural metric (default 1)\n"
  "  --early           also decode WSJT-X's early passes (default: answer them at once with\n"
  "                    no decodes, see README: a slow early pass costs the full one)\n"
  "  --max-decodes N   report at most N decodes per WSJT-X pass (default 200)\n"
  "  --no-range        report decodes outside WSJT-X's [nfa, nfb] too\n"
  "  -v, --verbose     log every pass to stderr (WSJT-X copies it into its log)\n"
  "\n"
  "Install: WSJTX_FT8_DECODER='/path/to/ft8nn_sidecar --model /path/to/ft8nn.onnx' wsjtx\n";

static bool parse_args (int argc, char ** argv, Options & o)
{
  for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];
      auto next = [&] () -> std::string {
        if (i + 1 >= argc)
          {
            say ("%s needs a value", a.c_str ());
            return {};
          }
        return argv[++i];
      };
      if (a == "--model") o.model = next ();
      else if (a == "--threads") o.threads = std::max (1, atoi (next ().c_str ()));
      else if (a == "--max-decodes") o.max_decodes = std::max (1, atoi (next ().c_str ()));
      else if (a == "--early") o.early = true;
      else if (a == "--no-range") o.range = false;
      else if (a == "-v" || a == "--verbose") verbose = true;
      else if (a == "-h" || a == "--help") o.help = true;
      else if (!a.empty () && a[0] == '-')
        {
          say ("unknown option %s (--help for usage)", a.c_str ());
          return false;
        }
      else o.files.push_back (a);
    }
  return true;
}

/* ---------------------------------------------------------------- model */

static bool readable (std::string const & p)
{
  return !p.empty () && access (p.c_str (), R_OK) == 0;
}

static std::string dir_of (std::string const & path)
{
  auto s = path.rfind ('/');
  return s == std::string::npos ? std::string {"."} : path.substr (0, s);
}

static std::string env_or_empty (char const * name)
{
  char const * e = getenv (name);
  return e && *e ? std::string {e} : std::string {};
}

/* --model, $FT8NN_MODEL, then ft8nn.onnx beside the binary, in the directories WSJT-X exports
 * (WSJTX_DATA_DIR holds ALL.TXT, WSJTX_SHARE_DIR the shipped data files), and in a source tree. */
static std::string find_model (Options const & o, char const * argv0)
{
  if (!o.model.empty ()) return o.model;
  if (std::string e = env_or_empty ("FT8NN_MODEL"); !e.empty ()) return e;
  std::string self = dir_of (argv0);
  for (std::string const & d : {self, env_or_empty ("WSJTX_DATA_DIR"), env_or_empty ("WSJTX_SHARE_DIR"),
                                self + "/../models/v3", self + "/../../models/v3"})
    {
      if (d.empty ()) continue;
      std::string p = d + "/ft8nn.onnx";
      if (readable (p)) return p;
    }
  return {};
}

/* ---------------------------------------------------------------- decoding */

/* The a-priori type WSJT-X shows after a decode: jt9 writes a1..a9 for its own hypotheses, and
 * ours are "the message starts with CQ" (a1), "the calls of a message this station sent 30 or
 * 60 s ago" (a2) and "exactly the message this station sent 30 or 60 s ago" (a3, the repeat
 * stage); jt9 has no equivalent of a2 or a3.  Everything else is a plain decode. */
static int ap_type (ft8nn_decode_t const & d)
{
  if (d.method == FT8NN_BP_AP || d.method == FT8NN_OSD_AP) return 1;
  if (d.method == FT8NN_BP_CTX || d.method == FT8NN_OSD_CTX) return 2;
  if (d.method == FT8NN_REPEAT) return 3;
  return 0;
}

struct Decoder
{
  ft8nn_ctx * ctx = nullptr;
  std::vector<float> audio;

  bool open (Options const & o, char const * argv0)
  {
    std::string model = find_model (o, argv0);
    if (model.empty ())
      {
        say ("no ft8nn.onnx found: pass --model PATH or set FT8NN_MODEL");
        return false;
      }
    ctx = ft8nn_open (model.c_str (), o.threads);
    if (!ctx)
      {
        say ("ft8nn_open failed for %s", model.c_str ());
        return false;
      }
    say ("model %s, %d ONNX thread(s)", model.c_str (), o.threads);
    audio.resize ((size_t) SLOT_SAMPLES);
    return true;
  }

  /* Decode n samples of audio and hand the decodes to sink(batch, pass) as the decoder finalises
   * them (ft8nn_decode_stream): once per subtraction pass, pass = 0, 1, ..., then once with pass = -1
   * and the context a-priori decodes, which have to wait for the last pass. Each batch keeps what
   * [nfa, nfb] allows, in frequency order like a jt9 pass; at most --max-decodes in all, counted in
   * the order the decoder finds them. slot_time is the period's start as epoch seconds, which the
   * decoder needs for its context a-priori decoding (it looks at this station's own decodes 30 and
   * 60 s earlier); INT64_MIN if it is not known. Returns the number reported, -1 if decoding failed. */
  template <class Sink>
  int run (Options const & o, int n, int64_t slot_time, int nfa, int nfb, Sink && sink)
  {
    struct State
    {
      Options const * o;
      int nfa, nfb, seen, reported;
      Sink * sink;
      std::vector<ft8nn_decode_t const *> keep;
    } st {&o, nfa, nfb, 0, 0, &sink, {}};
    ft8nn_pass_cb cb = [] (ft8nn_decode_t const * d, int nd, int pass, void * user) {
      auto & s = *static_cast<State *> (user);
      s.keep.clear ();
      for (int i = 0; i < nd && s.seen < s.o->max_decodes; ++i, ++s.seen)
        {
          if (s.o->range && s.nfb > s.nfa && s.nfa >= 0 && (d[i].f0 < s.nfa || d[i].f0 > s.nfb)) continue;
          s.keep.push_back (&d[i]);
        }
      std::sort (s.keep.begin (), s.keep.end (),
                 [] (ft8nn_decode_t const * a, ft8nn_decode_t const * b) { return a->f0 < b->f0; });
      s.reported += (int) s.keep.size ();
      (*s.sink) (s.keep, pass);
    };
    /* INT64_MIN is FT8NN_NO_SLOT_TIME: no context decoding, as ft8nn_decode */
    if (ft8nn_decode_stream (ctx, audio.data (), n, slot_time, cb, &st, nullptr, 0) < 0)
      {
        say ("decode failed");
        return -1;
      }
    return st.reported;
  }

  ~Decoder () { if (ctx) ft8nn_close (ctx); }
};

/* ---------------------------------------------------------------- sidecar replies */

static void send_decode (long long seq, ft8nn_decode_t const & d)
{
  /* WSJT-X puts these into fixed-width Fortran fields, so keep them inside what it can show. */
  int snr = std::min (99, std::max (-99, (int) lround (d.snr)));
  double dt = std::min (9.9, std::max (-9.9, (double) d.dt));
  int freq = std::min (99999, std::max (0, (int) lround (d.f0)));
  std::string msg = d.text;
  while (!msg.empty () && msg.back () == ' ') msg.pop_back ();
  printf ("{\"seq\":%lld,\"snr\":%d,\"dt\":%.1f,\"freq\":%d,\"msg\":%s", seq, snr, dt, freq,
          json_string (msg).c_str ());
  if (int ap = ap_type (d); ap) printf (",\"ap\":%d", ap);
  fputs ("}\n", stdout);
}

static void send_done (long long seq)
{
  printf ("{\"seq\":%lld,\"done\":true}\n", seq);
  fflush (stdout);
}

/* ---------------------------------------------------------------- slot time */

/* "2026-11-08T12:30:15.000Z" -> epoch seconds, or INT64_MIN. */
static int64_t parse_iso8601 (std::string const & s)
{
  struct tm tm;
  memset (&tm, 0, sizeof tm);
  int y = 0, mo = 0, d = 0, h = 0, mi = 0, se = 0;
  if (sscanf (s.c_str (), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &se) != 6) return INT64_MIN;
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || se > 60) return INT64_MIN;
  tm.tm_year = y - 1900;
  tm.tm_mon = mo - 1;
  tm.tm_mday = d;
  tm.tm_hour = h;
  tm.tm_min = mi;
  tm.tm_sec = se;
  return (int64_t) timegm (&tm);
}

/* The period's start, aligned to 15 s.  Live audio carries period_start (ISO-8601 UTC); audio from
 * a .wav file the user opened carries params.yymmdd and params.nutc instead, and when neither is
 * usable we fall back to today's date with params.nutc as the time of day. */
static int64_t slot_time_of (Json const & req)
{
  if (int64_t t = parse_iso8601 (req.str ("period_start")); t != INT64_MIN) return t - t % 15;

  int nutc = req.integer ("utc", req.integer ("params.nutc", -1));
  int hh = nutc / 10000, mm = (nutc / 100) % 100, ss = nutc % 100;
  if (nutc < 0 || hh > 23 || mm > 59 || ss > 59) return INT64_MIN;
  struct tm tm;
  memset (&tm, 0, sizeof tm);
  tm.tm_hour = hh;
  tm.tm_min = mm;
  tm.tm_sec = ss;
  int yymmdd = req.integer ("params.yymmdd", -1);
  if (yymmdd > 0)
    {
      tm.tm_year = 100 + yymmdd / 10000;
      tm.tm_mon = (yymmdd / 100) % 100 - 1;
      tm.tm_mday = yymmdd % 100;
      if (tm.tm_mon < 0 || tm.tm_mon > 11 || tm.tm_mday < 1) return INT64_MIN;
      int64_t t = (int64_t) timegm (&tm);
      return t - t % 15;
    }
  time_t now = time (nullptr);
  struct tm today;
  gmtime_r (&now, &today);
  tm.tm_year = today.tm_year;
  tm.tm_mon = today.tm_mon;
  tm.tm_mday = today.tm_mday;
  int64_t t = (int64_t) timegm (&tm);
  if (t - (int64_t) now > 43200) t -= 86400;      // just after UTC midnight
  if ((int64_t) now - t > 43200) t += 86400;      // just before it
  return t - t % 15;
}

/* ---------------------------------------------------------------- sidecar mode */

static int sidecar_mode (Options const & o, char const * argv0)
{
  Decoder dec;
  /* Without a model there is nothing we can do, and exiting is the useful answer: WSJT-X logs it,
   * says so in the status bar and goes back to its own jt9, instead of decoding nothing all day. */
  if (!dec.open (o, argv0)) return 1;

  LineReader in {STDIN_FILENO};
  std::vector<short> pcm;
  int have = 0;                                 // samples of the last request that carried audio
  int64_t have_slot = INT64_MIN;
  std::string line;

  for (;;)
    {
      auto r = in.next (line, 86400.);
      if (r == LineReader::Closed) { vsay ("stdin closed"); break; }
      if (r == LineReader::Error) { say ("stdin: %s", in.error ()); return 1; }
      if (r == LineReader::Timeout) continue;   // a day with nothing to do is not an error

      Json req;
      if (!req.parse (line))
        {
          say ("ignoring a line that is not JSON (%s)", req.error ());
          continue;
        }
      std::string type = req.str ("type");
      if (type == "quit") { vsay ("quit"); break; }
      if (type == "hello")
        {
          say ("hello from %s %s", req.str ("app", "?").c_str (), req.str ("version", "?").c_str ());
          continue;
        }
      if (type != "decode")
        {
          say ("ignoring a %s message", type.empty () ? "typeless" : type.c_str ());
          continue;
        }

      long long seq = req.i64 ("seq", 0);
      int nsamp = req.integer ("samples", 0);
      int rate = req.integer ("sample_rate", SAMPLE_RATE);

      /* The audio follows the line, so it has to come off the pipe whether we decode it or not. */
      if (nsamp < 0 || nsamp > 30 * 60 * SAMPLE_RATE)
        {
          say ("seq %lld: samples = %d is not a length of audio; stopping", seq, nsamp);
          send_done (seq);
          break;                                /* the stream is out of step: nothing else is safe */
        }
      if (nsamp > 0)
        {
          pcm.resize ((size_t) nsamp);
          auto ar = in.read_exact ((char *) pcm.data (), (size_t) nsamp * 2, 30.);
          if (ar != LineReader::Line)
            {
              say ("seq %lld: only part of the %d samples arrived (%s)", seq, nsamp,
                   ar == LineReader::Timeout ? "timeout" : ar == LineReader::Closed ? "stdin closed"
                   : in.error ());
              send_done (seq);
              break;
            }
        }

      std::string mode = req.str ("mode", "FT8");
      bool final_pass = req.flag ("final", true);
      int pass = req.integer ("pass", 0);
      int nfa = req.integer ("params.nfa", 0), nfb = req.integer ("params.nfb", 0);
      double budget = std::max (5., (double) req.integer ("params.ntrperiod", 15));

      if (rate != SAMPLE_RATE)
        {
          say ("seq %lld: %d Hz audio, this decoder needs %d Hz", seq, rate, SAMPLE_RATE);
          send_done (seq);
          continue;
        }
      if (mode != "FT8")
        {
          say ("seq %lld: %s is not ours to decode", seq, mode.c_str ());
          send_done (seq);
          continue;
        }

      if (nsamp > 0)
        {
          /* Keep the period's audio: a re-decode (the user clicking Decode) arrives with
           * samples = 0 and means "the last audio you were sent, again".  The bytes are
           * little-endian int16, which is also the host order on every platform WSJT-X ships on,
           * so they land in pcm[] as they are; the decoder wants them as floats on the same
           * int16 scale (that is what wav_read gives file mode). */
          have = std::min (nsamp, SLOT_SAMPLES);
          have_slot = slot_time_of (req);
          for (int i = 0; i < have; ++i) dec.audio[(size_t) i] = (float) pcm[(size_t) i];
          if (nsamp > SLOT_SAMPLES) vsay ("seq %lld: %d samples, using the first %d", seq, nsamp, have);
        }
      else if (!have)
        {
          say ("seq %lld: a re-decode, but no audio has been sent yet", seq);
          send_done (seq);
          continue;
        }

      /* WSJT-X asks for a decode at about 11.8 s and 13.2 s into the period as well as at the end,
       * and skips a pass that arrives while we are still busy - so a slow early pass costs the full
       * one. This decoder needs seconds per slot, so early passes are answered at once, with no
       * decodes; the full pass finds those signals anyway. */
      if (!final_pass && !o.early)
        {
          vsay ("seq %lld: early pass %d (%d samples) skipped", seq, pass, nsamp);
          send_done (seq);
          continue;
        }

      double t0 = monotonic ();
      int k = dec.run (o, have, have_slot, nfa, nfb, [&] (auto const & keep, int p) {
          for (auto * d : keep) send_decode (seq, *d);
          fflush (stdout);
          if (p >= 0) vsay ("seq %lld:   decoder pass %d: %d decode(s) sent at %.2f s", seq, p, (int) keep.size (), monotonic () - t0);
          else vsay ("seq %lld:   context decodes: %d sent at %.2f s", seq, (int) keep.size (), monotonic () - t0);
        });
      send_done (seq);
      double took = monotonic () - t0;
      vsay ("seq %lld: utc %06d pass %d (%d samples) -> %d decode(s) in %.1f s",
            seq, req.integer ("utc", 0) % 1000000, pass, have, std::max (k, 0), took);
      if (took > budget)
        say ("seq %lld: %.1f s is longer than WSJT-X waits (%.0f s), so this pass was dropped;"
             " see wsjtx/README.md on decode speed", seq, took, budget);
    }
  return 0;
}

/* ---------------------------------------------------------------- file mode */

/* jt9's FT8 output line, lib/decoder.f90 format 1001:
 *     (i6.6, i4, f5.1, i5, ' ~ ', 1x, a37, 1x, a2)
 * This is what WSJT-X builds from our JSON replies (Ft8Sidecar::format_jt9_line), and what the
 * tests compare, so file mode prints it too. */
static void print_jt9_line (int nutc, ft8nn_decode_t const & d)
{
  int snr = std::min (99, std::max (-99, (int) lround (d.snr)));
  double dt = std::min (9.9, std::max (-9.9, (double) d.dt));
  int f = std::min (99999, std::max (0, (int) lround (d.f0)));
  char annot[3] = {' ', ' ', 0};
  if (int ap = ap_type (d); ap) { annot[0] = 'a'; annot[1] = (char) ('0' + ap); }
  printf ("%06d%4d%5.1f%5d ~  %-37.37s %2s\n", nutc % 1000000, snr, dt, f, d.text, annot);
}

static int file_mode (Options const & o, char const * argv0)
{
  Decoder dec;
  if (!dec.open (o, argv0)) return 1;
  for (auto const & path : o.files)
    {
      int n = 0, rate = 0;
      float * x = wav_read (path.c_str (), &n, &rate);
      if (!x || rate != SAMPLE_RATE)
        {
          say ("%s: need mono 12 kHz PCM16/float32 WAV", path.c_str ());
          free (x);
          continue;
        }
      std::string base = path.substr (path.rfind ('/') == std::string::npos ? 0 : path.rfind ('/') + 1);
      if (auto dot = base.rfind ('.'); dot != std::string::npos) base = base.substr (0, dot);
      int64_t t = wav_slot_time (base.c_str ());
      int nutc = 0;
      if (t != INT64_MIN)
        {
          time_t tt = (time_t) t;
          struct tm tm;
          gmtime_r (&tt, &tm);
          nutc = tm.tm_hour * 10000 + tm.tm_min * 100 + tm.tm_sec;
        }
      n = std::min (n, SLOT_SAMPLES);
      std::copy (x, x + n, dec.audio.begin ());
      free (x);
      int k = dec.run (o, n, t, 0, 0, [&] (auto const & keep, int) {
          for (auto * d : keep) print_jt9_line (nutc, *d);
        });
      /* '<DecodeFinished>', 2i4, i9 (lib/decoder.f90 format 1010): WSJT-X writes this itself when
       * our "done" arrives, so file mode prints it only to look like a jt9 pass. */
      printf ("<DecodeFinished>%4d%4d%9d\n", 0, std::max (k, 0), 0);
      fflush (stdout);
    }
  return 0;
}

int main (int argc, char ** argv)
{
  Options o;
  if (!parse_args (argc, argv, o)) return 2;
  if (o.help)
    {
      fputs (usage_text, stdout);
      return 0;
    }
  setvbuf (stdout, nullptr, _IOLBF, 0);         // WSJT-X reads our replies line by line
  return o.files.empty () ? sidecar_mode (o, argv[0]) : file_mode (o, argv[0]);
}

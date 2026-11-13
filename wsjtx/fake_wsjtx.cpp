/* fake_wsjtx: WSJT-X's half of the FT8 sidecar protocol, for testing ft8nn_sidecar without WSJT-X.
 *
 * It does what MainWindow::startFt8Sidecar() and Decoder/Ft8Sidecar.cpp do: start the decoder as a
 * child process with pipes on its stdin and stdout, send "hello", then for each 15 s slot send a
 * "decode" line followed by the audio as raw s16le, read reply lines until {"seq":N,"done":true},
 * and finally send "quit" and close stdin.
 *
 *   fake_wsjtx [options] FILE.wav...
 *     --decoder CMD    the decoder and its arguments, one string (default "build/ft8nn_sidecar")
 *     --pass N         the pass number to report (nzhsym; default 50)
 *     --early          report final=false, as WSJT-X does for its 11.8 s and 13.5 s passes
 *     --sequence       per slot the three requests WSJT-X sends live: nzhsym 41 (11.8 s of audio) and 47
 *                      (13.5 s), final=false, then 50 (14.4 s), final=true
 *     --live           live audio: period_start set, disk=false (default: a .wav the user opened)
 *     --redecode       after each slot, a second request with samples=0 (the user clicked Decode)
 *     --garbage        send a junk line before each decode, to check it is ignored
 *     --nfa N --nfb N  the frequency range to decode (default 200..4000)
 *     --timeout S      how long to wait for "done" (default 120; WSJT-X waits 15)
 *     --times FILE     write each reply's arrival time to FILE: "seq<TAB>seconds<TAB>message" per decode
 *                      and "seq<TAB>seconds<TAB><done>", seconds since the request and its audio were
 *                      written - the latency WSJT-X sees, since it shows each decode line on arrival
 *
 * Every reply is formatted into the jt9 output line WSJT-X would build from it
 * (Ft8Sidecar::format_jt9_line), and each request ends with the "<DecodeFinished>" line WSJT-X
 * feeds to the rest of itself, so stdout here is exactly what WSJT-X's own decode handling sees.
 * Exit status is 1 if a request went unanswered, a reply carried the wrong seq, or the decoder had
 * to be killed.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "json.h"
#include "linereader.h"
#include "wav.h"

static int const SAMPLE_RATE = 12000;
static int const SLOT_SAMPLES = 15 * SAMPLE_RATE;

static void msleep (int ms)
{
  struct timespec ts = {ms / 1000, (long) (ms % 1000) * 1000000L};
  nanosleep (&ts, nullptr);
}

template<typename T> static T clamp_field (T v, T lo, T hi) { return std::max (lo, std::min (hi, v)); }

/* Ft8Sidecar::format_jt9_line: what WSJT-X builds from one reply line. */
static std::string format_jt9_line (int utc, int snr, double dt, int freq, std::string msg,
                                    int ap, bool lowconf)
{
  while (!msg.empty () && (msg.back () == ' ' || msg.back () == '\t')) msg.pop_back ();
  size_t lead = msg.find_first_not_of (" \t");
  msg = lead == std::string::npos ? std::string {} : msg.substr (lead);
  msg = msg.substr (0, 37);
  msg.resize (37, ' ');
  if (lowconf) msg[36] = '?';
  std::string annot = ap > 0 ? "a" + std::to_string (std::min (ap, 9)) : "  ";
  char head[64];
  snprintf (head, sizeof head, "%06d%4d%5.1f%5d ~  ", clamp_field (utc, 0, 999999),
            clamp_field (snr, -999, 9999), clamp_field (dt, -99.9, 999.9),
            clamp_field (freq, 0, 99999));
  return head + msg + ' ' + annot;
}

/* The command line is split on spaces, as QProcess::splitCommand would (without the quoting). */
static std::vector<std::string> split_command (std::string const & cmd)
{
  std::vector<std::string> out;
  size_t i = 0;
  while (i < cmd.size ())
    {
      while (i < cmd.size () && cmd[i] == ' ') ++i;
      size_t j = cmd.find (' ', i);
      if (j == std::string::npos) j = cmd.size ();
      if (j > i) out.push_back (cmd.substr (i, j - i));
      i = j;
    }
  return out;
}

static bool write_all (int fd, void const * data, size_t n)
{
  char const * p = (char const *) data;
  while (n)
    {
      ssize_t k = write (fd, p, n);
      if (k > 0) { p += k; n -= (size_t) k; continue; }
      if (k < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      return false;
    }
  return true;
}

struct Options
{
  std::string decoder = "build/ft8nn_sidecar";
  int pass = 50;
  bool early = false, live = false, redecode = false, garbage = false, sequence = false;
  int nfa = 200, nfb = 4000;
  double timeout = 120.;
  std::string times;
  std::vector<std::string> files;
};

static double monotonic ()
{
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main (int argc, char ** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];
      auto next = [&] () -> std::string {
        if (i + 1 >= argc) { fprintf (stderr, "%s needs a value\n", a.c_str ()); exit (2); }
        return argv[++i];
      };
      if (a == "--decoder") o.decoder = next ();
      else if (a == "--pass") o.pass = atoi (next ().c_str ());
      else if (a == "--early") o.early = true;
      else if (a == "--sequence") o.sequence = true;
      else if (a == "--live") o.live = true;
      else if (a == "--redecode") o.redecode = true;
      else if (a == "--garbage") o.garbage = true;
      else if (a == "--times") o.times = next ();
      else if (a == "--nfa") o.nfa = atoi (next ().c_str ());
      else if (a == "--nfb") o.nfb = atoi (next ().c_str ());
      else if (a == "--timeout") o.timeout = atof (next ().c_str ());
      else if (a == "-h" || a == "--help")
        {
          fputs ("fake_wsjtx [--decoder CMD] [--pass N] [--early] [--sequence] [--live] [--redecode] [--garbage]\n"
                 "           [--nfa N] [--nfb N] [--timeout S] [--times FILE] FILE.wav...\n", stdout);
          return 0;
        }
      else o.files.push_back (a);
    }
  if (o.files.empty ())
    {
      fputs ("fake_wsjtx: no .wav files given (--help for usage)\n", stderr);
      return 2;
    }

  signal (SIGPIPE, SIG_IGN);                    // a dead decoder is an error, not a signal

  /* Start the decoder, the way WSJT-X starts it: once, for the whole session. */
  auto args = split_command (o.decoder);
  if (args.empty ()) { fputs ("fake_wsjtx: --decoder is empty\n", stderr); return 2; }
  std::vector<char *> cargv;
  for (auto & a : args) cargv.push_back (const_cast<char *> (a.c_str ()));
  cargv.push_back (nullptr);
  int to_child[2], from_child[2];
  if (pipe (to_child) || pipe (from_child)) { perror ("pipe"); return 1; }
  pid_t pid = fork ();
  if (pid < 0) { perror ("fork"); return 1; }
  if (!pid)
    {
      dup2 (to_child[0], STDIN_FILENO);
      dup2 (from_child[1], STDOUT_FILENO);
      close (to_child[0]); close (to_child[1]);
      close (from_child[0]); close (from_child[1]);
      execv (cargv[0], cargv.data ());
      fprintf (stderr, "fake_wsjtx: cannot run %s: %s\n", cargv[0], strerror (errno));
      _exit (127);
    }
  close (to_child[0]);
  close (from_child[1]);
  int in_fd = to_child[1];
  LineReader reader {from_child[0]};

  std::string hello = "{\"type\":\"hello\",\"v\":1,\"app\":\"WSJT-X\",\"version\":\"3.0.0\","
    "\"temp_dir\":\".\",\"data_dir\":\".\",\"share_dir\":\".\"}\n";
  if (!write_all (in_fd, hello.data (), hello.size ()))
    {
      fputs ("fake_wsjtx: the decoder is not reading its stdin\n", stderr);
      return 1;
    }

  int failures = 0;
  FILE * times = o.times.empty () ? nullptr : fopen (o.times.c_str (), "w");
  if (!o.times.empty () && !times)
    {
      fprintf (stderr, "fake_wsjtx: cannot write %s: %s\n", o.times.c_str (), strerror (errno));
      return 2;
    }
  long long seq = 0;
  for (auto const & path : o.files)
    {
      int n = 0, rate = 0;
      float * x = wav_read (path.c_str (), &n, &rate);
      if (!x || rate != SAMPLE_RATE)
        {
          fprintf (stderr, "fake_wsjtx: %s: need mono 12 kHz PCM16/float32 WAV\n", path.c_str ());
          free (x);
          ++failures;
          continue;
        }
      n = std::min (n, SLOT_SAMPLES);
      std::vector<short> pcm ((size_t) n);
      for (int i = 0; i < n; ++i)
        pcm[(size_t) i] = (short) lround (std::min (32767.f, std::max (-32768.f, x[i])));
      free (x);

      /* The slot's clock time, from its yymmdd_HHMMSS file name. */
      std::string base = path.substr (path.rfind ('/') == std::string::npos ? 0 : path.rfind ('/') + 1);
      if (auto dot = base.rfind ('.'); dot != std::string::npos) base = base.substr (0, dot);
      int64_t t = wav_slot_time (base.c_str ());
      int nutc = 0, yymmdd = -1;
      char iso[40] = "";
      if (t != INT64_MIN)
        {
          time_t tt = (time_t) t;
          struct tm tm;
          gmtime_r (&tt, &tm);
          nutc = tm.tm_hour * 10000 + tm.tm_min * 100 + tm.tm_sec;
          yymmdd = (tm.tm_year - 100) * 10000 + (tm.tm_mon + 1) * 100 + tm.tm_mday;
          strftime (iso, sizeof iso, "%Y-%m-%dT%H:%M:%S.000Z", &tm);
        }

      struct Ask { int pass; bool final; int nsamp; };
      std::vector<Ask> asks;
      if (o.sequence)
        asks = {{41, false, std::min (n, 41 * 3456)}, {47, false, std::min (n, 47 * 3456)}, {50, true, std::min (n, 50 * 3456)}};
      else
        asks = {{o.pass, !o.early, n}};
      if (o.redecode) asks.push_back ({asks.back ().pass, asks.back ().final, 0});
      for (size_t attempt = 0; attempt < asks.size (); ++attempt)
        {
          bool again = asks[attempt].nsamp == 0;    // the user clicked Decode: no audio, same period
          int nsamp = asks[attempt].nsamp;
          int apass = asks[attempt].pass;
          bool afinal = asks[attempt].final;
          ++seq;
          if (o.garbage)
            {
              std::string junk = "not json at all\n";
              write_all (in_fd, junk.data (), junk.size ());
            }
          /* The protocol carries every dec_data.params field; these are the ones that matter for
           * FT8 (doc/ft8-sidecar-protocol.md), which is all this decoder reads. */
          char req[2048];
          int k = snprintf (req, sizeof req,
            "{\"type\":\"decode\",\"v\":1,\"seq\":%lld,\"mode\":\"FT8\",\"utc\":%d,\"pass\":%d,"
            "\"final\":%s,\"disk\":%s,\"period_start\":\"%s\",\"dial_freq\":14074000,"
            "\"special_op\":0,\"sample_rate\":%d,\"samples\":%d,"
            "\"params\":{\"nutc\":%d,\"ndiskdat\":%s,\"ntrperiod\":15,\"nQSOProgress\":0,"
            "\"nfqso\":1500,\"nftx\":1500,\"newdat\":%s,\"nagain\":%s,\"nfa\":%d,\"nfSplit\":2700,"
            "\"nfb\":%d,\"ntol\":50,\"kin\":%d,\"nzhsym\":%d,\"ndepth\":3,\"lft8apon\":true,"
            "\"lapcqonly\":false,\"napwid\":75,\"nmode\":8,\"ntxmode\":8,\"b_superfox\":false,"
            "\"b_even_seq\":%s,\"yymmdd\":%d,\"listutc\":[0,0,0,0,0,0,0,0,0,0],"
            "\"datetime\":\"2026-Nov-09 00:00\",\"mycall\":\"K1ABC\",\"mygrid\":\"FN42\","
            "\"hiscall\":\"\",\"hisgrid\":\"\",\"mybcall\":\"K1ABC\",\"hisbcall\":\"\"}}\n",
            seq, nutc, apass, afinal ? "true" : "false", o.live ? "false" : "true",
            o.live ? iso : "", SAMPLE_RATE, nsamp, nutc, o.live ? "false" : "true",
            again ? "false" : "true", again ? "true" : "false", o.nfa, o.nfb, again ? n : nsamp, apass,
            nutc % 30 < 15 ? "true" : "false", o.live ? -1 : yymmdd);
          if (k <= 0 || k >= (int) sizeof req)
            {
              fputs ("fake_wsjtx: the request did not fit\n", stderr);
              return 1;
            }
          if (!write_all (in_fd, req, (size_t) k)
              || (nsamp && !write_all (in_fd, pcm.data (), (size_t) nsamp * 2)))
            {
              fputs ("fake_wsjtx: the decoder stopped reading its stdin\n", stderr);
              ++failures;
              goto done;
            }

          /* Read the replies, as Ft8Sidecar::handle_line() does. */
          double t_sent = monotonic ();
          int ndecoded = 0;
          bool finished = false;
          std::string line;
          while (!finished)
            {
              auto r = reader.next (line, o.timeout);
              if (r == LineReader::Timeout)
                {
                  fprintf (stderr, "fake_wsjtx: seq %lld: no reply within %.0f s\n", seq, o.timeout);
                  ++failures;
                  break;
                }
              if (r != LineReader::Line)
                {
                  fprintf (stderr, "fake_wsjtx: the decoder stopped writing%s%s\n",
                           r == LineReader::Error ? ": " : "",
                           r == LineReader::Error ? reader.error () : "");
                  ++failures;
                  goto done;
                }
              Json reply;
              if (!reply.parse (line))
                {
                  fprintf (stderr, "fake_wsjtx: ignoring a non-JSON reply: %s", line.c_str ());
                  continue;                     // WSJT-X logs and ignores it
                }
              if (reply.has ("log"))
                {
                  fprintf (stderr, "fake_wsjtx: decoder log: %s\n", reply.str ("log").c_str ());
                  continue;
                }
              if (!reply.has ("seq"))
                {
                  fprintf (stderr, "fake_wsjtx: a reply without seq (WSJT-X allows it, but this"
                           " decoder should always echo it): %s", line.c_str ());
                  ++failures;
                }
              else if (reply.i64 ("seq") != seq)
                {
                  fprintf (stderr, "fake_wsjtx: a reply for seq %lld arrived during seq %lld;"
                           " WSJT-X would drop it\n", reply.i64 ("seq"), seq);
                  ++failures;
                  continue;
                }
              if (reply.flag ("done"))
                {
                  if (times) fprintf (times, "%lld\t%.3f\t<done>\n", seq, monotonic () - t_sent);
                  finished = true;
                  break;
                }
              if (!reply.has ("msg")) continue;
              ++ndecoded;
              if (times) fprintf (times, "%lld\t%.3f\t%s\n", seq, monotonic () - t_sent, reply.str ("msg").c_str ());
              puts (format_jt9_line (nutc, reply.integer ("snr"), reply.num ("dt"),
                                     (int) lround (reply.num ("freq")), reply.str ("msg"),
                                     reply.integer ("ap"), reply.flag ("lowconf")).c_str ());
            }
          /* What WSJT-X injects when "done" arrives: '<DecodeFinished>', 2i4, i9. */
          printf ("<DecodeFinished>%4d%4d%9d\n", 0, ndecoded, 0);
          fflush (stdout);
        }
    }

done:
  {
    std::string quit = "{\"type\":\"quit\",\"v\":1}\n";
    write_all (in_fd, quit.data (), quit.size ());
    close (in_fd);                              // and close stdin, as the protocol says
    bool gone = false;
    for (int i = 0; i < 50; ++i)                // "exit within 1 s or be killed"
      {
        int st;
        pid_t r = waitpid (pid, &st, WNOHANG);
        if (r == pid || (r < 0 && errno == ECHILD)) { gone = true; break; }
        msleep (20);
      }
    if (!gone)
      {
        fputs ("fake_wsjtx: the decoder did not exit within 1 s; killing it\n", stderr);
        kill (pid, SIGKILL);
        int st;
        waitpid (pid, &st, 0);
        ++failures;
      }
    close (from_child[0]);
  }
  if (times) fclose (times);
  return failures ? 1 : 0;
}

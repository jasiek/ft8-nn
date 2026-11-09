/* Read lines, and exact byte counts, from a pipe with a timeout.
 *
 * Deliberately not stdio: poll() reports what the *file descriptor* has, while fgets() buffers
 * whatever it managed to read, so a poll()/fgets() pair stalls with complete lines still sitting in
 * stdio's buffer. This reads with read() into a buffer of its own and hands out whole lines - or,
 * for the sidecar protocol's audio, the exact number of bytes that follow a line.
 */
#pragma once
#include <algorithm>
#include <string>

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

class LineReader
{
public:
  explicit LineReader (int fd) : fd_ {fd} {}

  enum Result { Line, Timeout, Closed, Error };

  /* A line including its '\n', or why not. Waits at most timeout_s in total. */
  Result next (std::string & line, double timeout_s)
  {
    Clock clock {timeout_s};
    for (;;)
      {
        auto nl = buf_.find ('\n');
        if (nl != std::string::npos)
          {
            line = buf_.substr (0, nl + 1);
            buf_.erase (0, nl + 1);
            return Line;
          }
        if (eof_)
          {
            if (buf_.empty ()) return Closed;
            line = buf_ + "\n";             // last line without a newline
            buf_.clear ();
            return Line;
          }
        if (Result r = pump (clock); r != Line) return r;
      }
  }

  /* Exactly n bytes, whatever was already buffered first. Short reads are never silent: Closed
   * means the writer went away mid-message, Timeout that it stopped for timeout_s. */
  Result read_exact (char * dst, size_t n, double timeout_s)
  {
    Clock clock {timeout_s};
    size_t got = 0;
    for (;;)
      {
        if (got < n && !buf_.empty ())
          {
            size_t k = std::min (n - got, buf_.size ());
            memcpy (dst + got, buf_.data (), k);
            buf_.erase (0, k);
            got += k;
          }
        if (got == n) return Line;
        if (eof_) return Closed;
        if (Result r = pump (clock); r != Line) return r;
      }
  }

  char const * error () const { return err_.c_str (); }

private:
  /* Seconds left of a budget, so one call can poll repeatedly without losing track of time. */
  struct Clock
  {
    explicit Clock (double budget) : budget_ {budget} { clock_gettime (CLOCK_MONOTONIC, &t0_); }

    double left () const
    {
      struct timespec t1;
      clock_gettime (CLOCK_MONOTONIC, &t1);
      double spent = (t1.tv_sec - t0_.tv_sec) + 1e-9 * (t1.tv_nsec - t0_.tv_nsec);
      return budget_ - spent;
    }

    double budget_;
    struct timespec t0_;
  };

  /* Wait for and append one chunk. Line means "something arrived or we were interrupted". */
  Result pump (Clock const & clock)
  {
    double left = clock.left ();
    if (left <= 0) return Timeout;
    int wait_ms = (int) (left * 1000);
    struct pollfd pfd = {fd_, POLLIN, 0};
    int r = poll (&pfd, 1, wait_ms > 3600000 ? 3600000 : wait_ms);   // in slices, so a long
                                                                     // wait cannot overflow
    if (r < 0)
      {
        if (errno == EINTR) return Line;
        err_ = strerror (errno);
        return Error;
      }
    if (!r) return Line;                    // nothing yet; the caller loops until the budget ends
    char chunk[16384];
    ssize_t k = read (fd_, chunk, sizeof chunk);
    if (k > 0) buf_.append (chunk, (size_t) k);
    else if (!k) eof_ = true;
    else if (errno != EINTR && errno != EAGAIN)
      {
        err_ = strerror (errno);
        return Error;
      }
    return Line;
  }

  int fd_;
  std::string buf_, err_;
  bool eof_ = false;
};

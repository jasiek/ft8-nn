/* A small JSON reader and writer, enough for WSJT-X's FT8 sidecar protocol: one compact object per
 * line, no arrays we care about, no documents larger than a few kB.  Rather than pull in a JSON
 * library (or Qt) for that, this parses an object into a flat key -> value map:
 *
 *     {"type":"decode","seq":7,"params":{"nfa":200}}
 *       ->  type = "decode", seq = 7, params.nfa = 200
 *
 * Nested objects are flattened with '.'; arrays are parsed and dropped (the protocol's only array,
 * params.listutc, is a JT65 message list that FT8 has no use for).
 */
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

class Json
{
public:
  /* Parse one JSON object, trailing whitespace allowed.  False (and error()) if it is malformed. */
  bool parse (std::string const & text)
  {
    v_.clear ();
    err_.clear ();
    s_ = &text;
    i_ = 0;
    if (!object (std::string {}, true)) return false;
    ws ();
    if (i_ != text.size ()) return fail ("trailing text after the object");
    return true;
  }

  bool has (std::string const & key) const { return v_.find (key) != v_.end (); }

  double num (std::string const & key, double def = 0) const
  {
    auto it = v_.find (key);
    if (it == v_.end ()) return def;
    if (it->second.kind == Value::Num) return it->second.n;
    if (it->second.kind == Value::Bool) return it->second.b ? 1 : 0;   /* a Fortran logical */
    return def;
  }

  long long i64 (std::string const & key, long long def = 0) const
  {
    return (long long) llround (num (key, (double) def));
  }

  int integer (std::string const & key, int def = 0) const { return (int) i64 (key, def); }

  bool flag (std::string const & key, bool def = false) const
  {
    auto it = v_.find (key);
    if (it == v_.end ()) return def;
    if (it->second.kind == Value::Bool) return it->second.b;
    if (it->second.kind == Value::Num) return it->second.n != 0;
    return def;
  }

  std::string str (std::string const & key, std::string const & def = {}) const
  {
    auto it = v_.find (key);
    return it != v_.end () && it->second.kind == Value::Str ? it->second.s : def;
  }

  char const * error () const { return err_.c_str (); }

private:
  struct Value
  {
    enum Kind { Null, Bool, Num, Str } kind = Null;
    bool b = false;
    double n = 0;
    std::string s;
  };

  bool fail (char const * why)
  {
    if (err_.empty ()) err_ = why;
    return false;
  }

  char c () const { return (*s_)[i_]; }
  void ws () { while (i_ < s_->size () && (c () == ' ' || c () == '\t' || c () == '\r' || c () == '\n')) ++i_; }

  bool eat (char ch)
  {
    ws ();
    if (i_ < s_->size () && c () == ch) { ++i_; return true; }
    return false;
  }

  bool object (std::string const & prefix, bool store)
  {
    if (!eat ('{')) return fail ("expected '{'");
    if (eat ('}')) return true;
    for (;;)
      {
        std::string key;
        if (!string (key)) return false;
        if (!eat (':')) return fail ("expected ':'");
        if (!value (prefix + key, store)) return false;
        if (eat (',')) continue;
        if (eat ('}')) return true;
        return fail ("expected ',' or '}'");
      }
  }

  bool array ()
  {
    if (!eat ('[')) return fail ("expected '['");
    if (eat (']')) return true;
    for (;;)
      {
        if (!value (std::string {}, false)) return false;
        if (eat (',')) continue;
        if (eat (']')) return true;
        return fail ("expected ',' or ']'");
      }
  }

  bool value (std::string const & key, bool store)
  {
    ws ();
    if (i_ >= s_->size ()) return fail ("unexpected end of input");
    char ch = c ();
    if (ch == '{') return object (store ? key + "." : key, store);
    if (ch == '[') return array ();
    Value val;
    if (ch == '"')
      {
        if (!string (val.s)) return false;
        val.kind = Value::Str;
      }
    else if (s_->compare (i_, 4, "true") == 0) { i_ += 4; val.kind = Value::Bool; val.b = true; }
    else if (s_->compare (i_, 5, "false") == 0) { i_ += 5; val.kind = Value::Bool; val.b = false; }
    else if (s_->compare (i_, 4, "null") == 0) { i_ += 4; val.kind = Value::Null; }
    else
      {
        char const * p = s_->c_str () + i_;
        char * end = nullptr;
        double d = strtod (p, &end);
        if (end == p) return fail ("not a JSON value");
        i_ += (size_t) (end - p);
        val.kind = Value::Num;
        val.n = d;
      }
    if (store) v_[key] = val;
    return true;
  }

  bool string (std::string & out)
  {
    ws ();
    if (i_ >= s_->size () || c () != '"') return fail ("expected a string");
    ++i_;
    out.clear ();
    while (i_ < s_->size ())
      {
        char ch = (*s_)[i_++];
        if (ch == '"') return true;
        if (ch != '\\') { out += ch; continue; }
        if (i_ >= s_->size ()) break;
        switch (char e = (*s_)[i_++]; e)
          {
          case '"': case '\\': case '/': out += e; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u':
            {
              if (i_ + 4 > s_->size ()) return fail ("truncated \\u escape");
              unsigned cp = (unsigned) strtoul (s_->substr (i_, 4).c_str (), nullptr, 16);
              i_ += 4;
              /* UTF-8 for the basic plane; this protocol never sends surrogate pairs. */
              if (cp < 0x80) out += (char) cp;
              else if (cp < 0x800)
                {
                  out += (char) (0xc0 | (cp >> 6));
                  out += (char) (0x80 | (cp & 0x3f));
                }
              else
                {
                  out += (char) (0xe0 | (cp >> 12));
                  out += (char) (0x80 | ((cp >> 6) & 0x3f));
                  out += (char) (0x80 | (cp & 0x3f));
                }
              break;
            }
          default: return fail ("bad escape in a string");
          }
      }
    return fail ("unterminated string");
  }

  std::map<std::string, Value> v_;
  std::string err_;
  std::string const * s_ = nullptr;
  size_t i_ = 0;
};

/* A JSON string literal, quotes and escapes included. */
inline std::string json_string (std::string const & s)
{
  std::string out = "\"";
  for (unsigned char ch : s)
    {
      switch (ch)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
          if (ch < 0x20)
            {
              char b[8];
              snprintf (b, sizeof b, "\\u%04x", ch);
              out += b;
            }
          else out += (char) ch;
        }
    }
  return out + "\"";
}

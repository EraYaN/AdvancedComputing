#pragma once
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <cstring>
#include <string>

std::string string_format(const std::string fmt_str, ...);
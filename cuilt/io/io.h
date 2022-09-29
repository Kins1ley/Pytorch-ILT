#ifndef IO_H_
#define IO_H_

#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include "google/protobuf/message.h"
#include "proto/cuilt.pb.h"

using ::google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const std::string& filename, Message* proto) {
    return ReadProtoFromTextFile(filename.c_str(), proto);
}
#endif
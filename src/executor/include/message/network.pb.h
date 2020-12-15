// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: network.proto

#ifndef PROTOBUF_network_2eproto__INCLUDED
#define PROTOBUF_network_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace com {
namespace nus {
namespace dbsytem {
namespace falcon {
namespace v0 {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_network_2eproto();
void protobuf_AssignDesc_network_2eproto();
void protobuf_ShutdownFile_network_2eproto();

class NetworkConfig;
class PortArray;

// ===================================================================

class NetworkConfig : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:com.nus.dbsytem.falcon.v0.NetworkConfig) */ {
 public:
  NetworkConfig();
  virtual ~NetworkConfig();

  NetworkConfig(const NetworkConfig& from);

  inline NetworkConfig& operator=(const NetworkConfig& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const NetworkConfig& default_instance();

  void Swap(NetworkConfig* other);

  // implements Message ----------------------------------------------

  inline NetworkConfig* New() const { return New(NULL); }

  NetworkConfig* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const NetworkConfig& from);
  void MergeFrom(const NetworkConfig& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(NetworkConfig* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated string ips = 1;
  int ips_size() const;
  void clear_ips();
  static const int kIpsFieldNumber = 1;
  const ::std::string& ips(int index) const;
  ::std::string* mutable_ips(int index);
  void set_ips(int index, const ::std::string& value);
  void set_ips(int index, const char* value);
  void set_ips(int index, const char* value, size_t size);
  ::std::string* add_ips();
  void add_ips(const ::std::string& value);
  void add_ips(const char* value);
  void add_ips(const char* value, size_t size);
  const ::google::protobuf::RepeatedPtrField< ::std::string>& ips() const;
  ::google::protobuf::RepeatedPtrField< ::std::string>* mutable_ips();

  // repeated .com.nus.dbsytem.falcon.v0.PortArray port_arrays = 2;
  int port_arrays_size() const;
  void clear_port_arrays();
  static const int kPortArraysFieldNumber = 2;
  const ::com::nus::dbsytem::falcon::v0::PortArray& port_arrays(int index) const;
  ::com::nus::dbsytem::falcon::v0::PortArray* mutable_port_arrays(int index);
  ::com::nus::dbsytem::falcon::v0::PortArray* add_port_arrays();
  ::google::protobuf::RepeatedPtrField< ::com::nus::dbsytem::falcon::v0::PortArray >*
      mutable_port_arrays();
  const ::google::protobuf::RepeatedPtrField< ::com::nus::dbsytem::falcon::v0::PortArray >&
      port_arrays() const;

  // @@protoc_insertion_point(class_scope:com.nus.dbsytem.falcon.v0.NetworkConfig)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::RepeatedPtrField< ::std::string> ips_;
  ::google::protobuf::RepeatedPtrField< ::com::nus::dbsytem::falcon::v0::PortArray > port_arrays_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_network_2eproto();
  friend void protobuf_AssignDesc_network_2eproto();
  friend void protobuf_ShutdownFile_network_2eproto();

  void InitAsDefaultInstance();
  static NetworkConfig* default_instance_;
};
// -------------------------------------------------------------------

class PortArray : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:com.nus.dbsytem.falcon.v0.PortArray) */ {
 public:
  PortArray();
  virtual ~PortArray();

  PortArray(const PortArray& from);

  inline PortArray& operator=(const PortArray& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const PortArray& default_instance();

  void Swap(PortArray* other);

  // implements Message ----------------------------------------------

  inline PortArray* New() const { return New(NULL); }

  PortArray* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const PortArray& from);
  void MergeFrom(const PortArray& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(PortArray* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int32 ports = 1;
  int ports_size() const;
  void clear_ports();
  static const int kPortsFieldNumber = 1;
  ::google::protobuf::int32 ports(int index) const;
  void set_ports(int index, ::google::protobuf::int32 value);
  void add_ports(::google::protobuf::int32 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      ports() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_ports();

  // @@protoc_insertion_point(class_scope:com.nus.dbsytem.falcon.v0.PortArray)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > ports_;
  mutable int _ports_cached_byte_size_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_network_2eproto();
  friend void protobuf_AssignDesc_network_2eproto();
  friend void protobuf_ShutdownFile_network_2eproto();

  void InitAsDefaultInstance();
  static PortArray* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// NetworkConfig

// repeated string ips = 1;
inline int NetworkConfig::ips_size() const {
  return ips_.size();
}
inline void NetworkConfig::clear_ips() {
  ips_.Clear();
}
inline const ::std::string& NetworkConfig::ips(int index) const {
  // @@protoc_insertion_point(field_get:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
  return ips_.Get(index);
}
inline ::std::string* NetworkConfig::mutable_ips(int index) {
  // @@protoc_insertion_point(field_mutable:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
  return ips_.Mutable(index);
}
inline void NetworkConfig::set_ips(int index, const ::std::string& value) {
  // @@protoc_insertion_point(field_set:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
  ips_.Mutable(index)->assign(value);
}
inline void NetworkConfig::set_ips(int index, const char* value) {
  ips_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
}
inline void NetworkConfig::set_ips(int index, const char* value, size_t size) {
  ips_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
}
inline ::std::string* NetworkConfig::add_ips() {
  // @@protoc_insertion_point(field_add_mutable:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
  return ips_.Add();
}
inline void NetworkConfig::add_ips(const ::std::string& value) {
  ips_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
}
inline void NetworkConfig::add_ips(const char* value) {
  ips_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
}
inline void NetworkConfig::add_ips(const char* value, size_t size) {
  ips_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
}
inline const ::google::protobuf::RepeatedPtrField< ::std::string>&
NetworkConfig::ips() const {
  // @@protoc_insertion_point(field_list:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
  return ips_;
}
inline ::google::protobuf::RepeatedPtrField< ::std::string>*
NetworkConfig::mutable_ips() {
  // @@protoc_insertion_point(field_mutable_list:com.nus.dbsytem.falcon.v0.NetworkConfig.ips)
  return &ips_;
}

// repeated .com.nus.dbsytem.falcon.v0.PortArray port_arrays = 2;
inline int NetworkConfig::port_arrays_size() const {
  return port_arrays_.size();
}
inline void NetworkConfig::clear_port_arrays() {
  port_arrays_.Clear();
}
inline const ::com::nus::dbsytem::falcon::v0::PortArray& NetworkConfig::port_arrays(int index) const {
  // @@protoc_insertion_point(field_get:com.nus.dbsytem.falcon.v0.NetworkConfig.port_arrays)
  return port_arrays_.Get(index);
}
inline ::com::nus::dbsytem::falcon::v0::PortArray* NetworkConfig::mutable_port_arrays(int index) {
  // @@protoc_insertion_point(field_mutable:com.nus.dbsytem.falcon.v0.NetworkConfig.port_arrays)
  return port_arrays_.Mutable(index);
}
inline ::com::nus::dbsytem::falcon::v0::PortArray* NetworkConfig::add_port_arrays() {
  // @@protoc_insertion_point(field_add:com.nus.dbsytem.falcon.v0.NetworkConfig.port_arrays)
  return port_arrays_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::com::nus::dbsytem::falcon::v0::PortArray >*
NetworkConfig::mutable_port_arrays() {
  // @@protoc_insertion_point(field_mutable_list:com.nus.dbsytem.falcon.v0.NetworkConfig.port_arrays)
  return &port_arrays_;
}
inline const ::google::protobuf::RepeatedPtrField< ::com::nus::dbsytem::falcon::v0::PortArray >&
NetworkConfig::port_arrays() const {
  // @@protoc_insertion_point(field_list:com.nus.dbsytem.falcon.v0.NetworkConfig.port_arrays)
  return port_arrays_;
}

// -------------------------------------------------------------------

// PortArray

// repeated int32 ports = 1;
inline int PortArray::ports_size() const {
  return ports_.size();
}
inline void PortArray::clear_ports() {
  ports_.Clear();
}
inline ::google::protobuf::int32 PortArray::ports(int index) const {
  // @@protoc_insertion_point(field_get:com.nus.dbsytem.falcon.v0.PortArray.ports)
  return ports_.Get(index);
}
inline void PortArray::set_ports(int index, ::google::protobuf::int32 value) {
  ports_.Set(index, value);
  // @@protoc_insertion_point(field_set:com.nus.dbsytem.falcon.v0.PortArray.ports)
}
inline void PortArray::add_ports(::google::protobuf::int32 value) {
  ports_.Add(value);
  // @@protoc_insertion_point(field_add:com.nus.dbsytem.falcon.v0.PortArray.ports)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
PortArray::ports() const {
  // @@protoc_insertion_point(field_list:com.nus.dbsytem.falcon.v0.PortArray.ports)
  return ports_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
PortArray::mutable_ports() {
  // @@protoc_insertion_point(field_mutable_list:com.nus.dbsytem.falcon.v0.PortArray.ports)
  return &ports_;
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace v0
}  // namespace falcon
}  // namespace dbsytem
}  // namespace nus
}  // namespace com

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_network_2eproto__INCLUDED
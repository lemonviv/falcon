// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: lr_grpc.proto
#ifndef GRPC_lr_5fgrpc_2eproto__INCLUDED
#define GRPC_lr_5fgrpc_2eproto__INCLUDED

#include "lr_grpc.pb.h"

#include <functional>
#include <grpc/impl/codegen/port_platform.h>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace com {
namespace nus {
namespace dbsytem {
namespace falcon {
namespace v0 {
namespace inference {

// The logistic regression inference service definition.
class InferenceLR final {
 public:
  static constexpr char const* service_full_name() {
    return "com.nus.dbsytem.falcon.v0.inference.InferenceLR";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // Sends a greeting
    virtual ::grpc::Status Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>> AsyncPrediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>>(AsyncPredictionRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>> PrepareAsyncPrediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>>(PrepareAsyncPredictionRaw(context, request, cq));
    }
    class experimental_async_interface {
     public:
      virtual ~experimental_async_interface() {}
      // Sends a greeting
      virtual void Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response, std::function<void(::grpc::Status)>) = 0;
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      virtual void Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      #else
      virtual void Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      #endif
    };
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    typedef class experimental_async_interface async_interface;
    #endif
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    async_interface* async() { return experimental_async(); }
    #endif
    virtual class experimental_async_interface* experimental_async() { return nullptr; }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* AsyncPredictionRaw(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* PrepareAsyncPredictionRaw(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>> AsyncPrediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>>(AsyncPredictionRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>> PrepareAsyncPrediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>>(PrepareAsyncPredictionRaw(context, request, cq));
    }
    class experimental_async final :
      public StubInterface::experimental_async_interface {
     public:
      void Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response, std::function<void(::grpc::Status)>) override;
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      void Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response, ::grpc::ClientUnaryReactor* reactor) override;
      #else
      void Prediction(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      #endif
     private:
      friend class Stub;
      explicit experimental_async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class experimental_async_interface* experimental_async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class experimental_async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* AsyncPredictionRaw(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* PrepareAsyncPredictionRaw(::grpc::ClientContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_Prediction_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    // Sends a greeting
    virtual ::grpc::Status Prediction(::grpc::ServerContext* context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_Prediction : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_Prediction() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_Prediction() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Prediction(::grpc::ServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestPrediction(::grpc::ServerContext* context, ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::grpc::ServerAsyncResponseWriter< ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Prediction<Service > AsyncService;
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_Prediction : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithCallbackMethod_Prediction() {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::Service::
    #else
      ::grpc::Service::experimental().
    #endif
        MarkMethodCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>(
            [this](
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
                   ::grpc::CallbackServerContext*
    #else
                   ::grpc::experimental::CallbackServerContext*
    #endif
                     context, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* request, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* response) { return this->Prediction(context, request, response); }));}
    void SetMessageAllocatorFor_Prediction(
        ::grpc::experimental::MessageAllocator< ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* allocator) {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
    #else
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::experimental().GetHandler(0);
    #endif
      static_cast<::grpc::internal::CallbackUnaryHandler< ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~ExperimentalWithCallbackMethod_Prediction() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Prediction(::grpc::ServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    virtual ::grpc::ServerUnaryReactor* Prediction(
      ::grpc::CallbackServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/)
    #else
    virtual ::grpc::experimental::ServerUnaryReactor* Prediction(
      ::grpc::experimental::CallbackServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/)
    #endif
      { return nullptr; }
  };
  #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
  typedef ExperimentalWithCallbackMethod_Prediction<Service > CallbackService;
  #endif

  typedef ExperimentalWithCallbackMethod_Prediction<Service > ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_Prediction : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_Prediction() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_Prediction() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Prediction(::grpc::ServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_Prediction : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_Prediction() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_Prediction() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Prediction(::grpc::ServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestPrediction(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_Prediction : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithRawCallbackMethod_Prediction() {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::Service::
    #else
      ::grpc::Service::experimental().
    #endif
        MarkMethodRawCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
                   ::grpc::CallbackServerContext*
    #else
                   ::grpc::experimental::CallbackServerContext*
    #endif
                     context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->Prediction(context, request, response); }));
    }
    ~ExperimentalWithRawCallbackMethod_Prediction() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Prediction(::grpc::ServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    virtual ::grpc::ServerUnaryReactor* Prediction(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)
    #else
    virtual ::grpc::experimental::ServerUnaryReactor* Prediction(
      ::grpc::experimental::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)
    #endif
      { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Prediction : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_Prediction() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler<
          ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* streamer) {
                       return this->StreamedPrediction(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_Prediction() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Prediction(::grpc::ServerContext* /*context*/, const ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest* /*request*/, ::com::nus::dbsytem::falcon::v0::inference::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedPrediction(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::com::nus::dbsytem::falcon::v0::inference::PredictionRequest,::com::nus::dbsytem::falcon::v0::inference::PredictionResponse>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_Prediction<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_Prediction<Service > StreamedService;
};

}  // namespace inference
}  // namespace v0
}  // namespace falcon
}  // namespace dbsytem
}  // namespace nus
}  // namespace com


#endif  // GRPC_lr_5fgrpc_2eproto__INCLUDED
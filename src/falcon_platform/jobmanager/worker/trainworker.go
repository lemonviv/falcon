package worker

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"falcon_platform/common"
	v0 "falcon_platform/common/proto/v0"
	"falcon_platform/jobmanager/base"
	"falcon_platform/jobmanager/entity"
	"falcon_platform/logger"
	"falcon_platform/resourcemanager"
	"github.com/golang/protobuf/proto"
	// 	"falcon_platform/utils"
	"net/rpc"
)

type TrainWorker struct {
	base.WorkerBase
}

/**
 * @Description
 * @Date 下午4:38 22/08/21
 * @Param
	masterAddr: master ip+port, worker send message to master
	workerAddr: current worker address, worker register it to master, receive master's call，
	PartyID: id of the party which launch this worker
	workerID： id of the worker, mainly used to distinguish workers in distributed training
	distributedRole： 0: ps, 1: worker
 * @return
 **/
func InitTrainWorker(masterAddr, workerAddr string,
	PartyID common.PartyIdType, WorkerID common.WorkerIdType, GroupID common.GroupIdType,
	DistributedRole uint) *TrainWorker {

	wk := TrainWorker{}
	wk.InitWorkerBase(workerAddr, common.TrainWorker)
	wk.MasterAddr = masterAddr
	wk.PartyID = PartyID
	wk.GroupID = GroupID
	wk.WorkerID = WorkerID
	wk.DistributedRole = DistributedRole

	logger.Log.Printf("[TrainWorker] Worker initialized, WorkerName=%s, WorkerID=%d, PartyID=%d, "+
		"DistributedRole=%s \n", wk.Name, wk.WorkerID, wk.PartyID, common.DistributedRoleToName(wk.DistributedRole))

	return &wk
}

/**
 * @Description Run worker
 * @Date 下午4:40 22/08/21
 * @Param
 * @return
 **/
func (wk *TrainWorker) Run() {

	// 0 thread: start event Loop
	go func() {
		defer logger.HandleErrors()
		wk.HeartBeatLoop()
	}()

	rpcSvc := rpc.NewServer()

	err := rpcSvc.Register(wk)
	if err != nil {
		logger.Log.Fatalf("%s: start Error \n", wk.Name)
	}

	logger.Log.Printf("[%s] from PartyID %d to register to masterAddr(%s)\n", wk.Name, wk.PartyID, wk.MasterAddr)
	wk.RegisterToMaster(wk.MasterAddr)

	// start rpc server blocking...
	wk.StartRPCServer(rpcSvc, true)

}

/**
 * @Description Called by master
 * @Date 下午4:40 22/08/21
 * @Param
 * @return
 **/
func (wk *TrainWorker) DoTask(args string, rep *entity.DoTaskReply) error {
	// 1. decode args

	taskArg, err := entity.DecodeGeneralTask([]byte(args))
	if err != nil {
		panic("Parser doTask args fails")
	}
	taskName := taskArg.TaskName
	algCfg := taskArg.AlgCfg

	logger.Log.Println("[TrainWorker] TrainWorker.DoTask called, taskName:", taskName)
	defer func() {
		logger.Log.Printf("[TrainWorker]: WorkerAddr %s, TrainWorker.DoTask Done for task %s \n", wk.Addr, taskName)
	}()

	// 2. init task manager for this task
	wk.Tm = resourcemanager.InitResourceManager()
	defer func() {
		// once all tasks finish, DeleteResources
		wk.Tm.DeleteResources()
		// clear wk.Tm
		wk.Tm = resourcemanager.InitResourceManager()
	}()

	update := func() {
		// update task error msg
		rep.TaskMsg.RuntimeMsg = wk.Tm.RunTimeErrorLog
		// update task status
		wk.Tm.Mux.Lock()
		if wk.Tm.TaskStatus == common.TaskFailed {
			rep.RuntimeError = true
		} else {
			rep.RuntimeError = false
		}
		wk.Tm.Mux.Unlock()
	}

	if taskName == common.PreProcSubTask {
		wk.printParams(wk.DslObj.Tasks.PreProcessing.AlgorithmName)
		wk.RunPreProcessingTask()
		// update task error msg
		update()
		return nil
	} else if taskName == common.ModelTrainSubTask {
		wk.printParams(wk.DslObj.Tasks.ModelTraining.AlgorithmName)
		wk.RunModelTrainingTask()
		// update task error msg
		update()
		return nil
	} else if taskName == common.LimeInstanceSampleTask {
		wk.printParams(wk.DslObj.Tasks.LimeInsSample.AlgorithmName)
		wk.RunLimeInstanceSampleTask()
		// update task error msg
		update()
		return nil
	} else if taskName == common.LimePredSubTask {
		wk.printParams(wk.DslObj.Tasks.LimePred.AlgorithmName)
		wk.RunLimePredTask()
		// update task error msg
		update()
		return nil
	} else if taskName == common.LimeWeightSubTask {
		wk.printParams(wk.DslObj.Tasks.LimeWeight.AlgorithmName)
		wk.RunLimeWeightTask()
		// update task error msg
		update()
		return nil
	} else if taskName == common.LimeFeatureSubTask {
		// update config
		wk.DslObj.Tasks.LimeFeature.InputConfigs.SerializedAlgorithmConfig = algCfg
		wk.printParams(wk.DslObj.Tasks.LimeFeature.AlgorithmName)
		wk.RunLimeFeatureTask()
		// update task error msg
		update()
		return nil
	} else if taskName == common.LimeInterpretSubTask {
		wk.DslObj.Tasks.LimeInterpret.InputConfigs.SerializedAlgorithmConfig = algCfg
		wk.printParams(wk.DslObj.Tasks.LimeInterpret.AlgorithmName)
		wk.RunLimeInterpretTask()
		// update task error msg
		update()
		return nil
	} else {
		rep.RuntimeError = true
		rep.TaskMsg.RuntimeMsg = "TaskName wrong! only support <pre_processing, model_training," +
			" lime_pred_task, lime_weight_task, lime_feature_task, lime_interpret_task>"
		panic("task name not found error, taskName=" + taskName)
	}

	return nil
}

func (wk *TrainWorker) RunMpc(mpcAlgorithmName string, rep *entity.DoTaskReply) error {

	//if wk.DslObj.DistributedTask.Enable == 1 && wk.DistributedRole == common.DistributedParameterServer {
	//	logger.Log.Println("[TrainWorker]: this is Parameter server, skip executing mpc")
	//	rep.RuntimeError = false
	//	return nil
	//}

	// 1. decode args
	logger.Log.Println("[TrainWorker] TrainWorker.RunMpc called, mpcAlgorithmName:", mpcAlgorithmName)
	defer func() {
		logger.Log.Printf("[TrainWorker]: WorkerAddr %s, TrainWorker.RunMpc for task %s done\n", wk.Addr, mpcAlgorithmName)
	}()

	update := func() {
		// update task error msg
		rep.TaskMsg.RuntimeMsg = wk.MpcTm.RunTimeErrorLog
		// update task status
		wk.MpcTm.Mux.Lock()
		if wk.MpcTm.TaskStatus == common.TaskFailed {
			rep.RuntimeError = true
		} else {
			rep.RuntimeError = false
		}
		wk.MpcTm.Mux.Unlock()
	}

	// clear the existing mpc resources if exist
	wk.MpcTm.DeleteResources()
	// init new resource manager
	wk.MpcTm = resourcemanager.InitResourceManager()
	// as for mpc, don't record it;s final status, as longs as it running, return as successful
	wk.RunMpcTask(mpcAlgorithmName)
	update()
	return nil
}

func (w *TrainWorker) RetrieveModelReport(_ string, rep *entity.RetrieveModelReportReply) error {
	rep.RuntimeError = false
	rep.ContainsModelReport = false

	return nil

	//workerActiveType := w.DslObj.PartyInfo.PartyType
	//
	//if workerActiveType == common.ActiveParty {
	//	rep.RuntimeError = false
	//	rep.ContainsModelReport = true
	//
	//	// read local file and send back
	//	//modelReportFile := common.TaskModelPath + "/" + w.DslObj.Tasks.ModelTraining.OutputConfigs.EvaluationReport
	//	modelReportFile := "/Users/nailixing/GOProj/src/github.com/scripts/report"
	//	logger.Log.Println("[TrainWorker] TrainWorker.RetrieveModelReport called, read file: ", modelReportFile)
	//	reportStr, err := utils.ReadFile(modelReportFile)
	//	if err != nil {
	//		rep.RuntimeError = true
	//		rep.TaskMsg.RuntimeMsg = err.Error()
	//	}
	//	rep.ModelReport = reportStr
	//} else if workerActiveType == common.PassiveParty {
	//	rep.RuntimeError = false
	//	rep.ContainsModelReport = false
	//} else {
	//	rep.RuntimeError = true
	//	rep.TaskMsg.RuntimeMsg = "PartyType not recognized"
	//}
	//logger.Log.Println("[TrainWorker] :RetrieveModelReport finished")
	//return nil
}

func (w *TrainWorker) printParams(algName string) {
	logger.Log.Println("[TrainWorker]: ----------- WorkerAddr , printParams -----------", algName)
	//return

	if algName == "" {

	} else if algName == common.LogisticRegressAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.ModelTraining.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LogisticRegressionParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.DecisionTreeAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.ModelTraining.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.DecisionTreeParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.RandomForestAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.ModelTraining.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.RandomForestParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

		logger.Log.Printf("[TrainWorker]: original DtParam = %+v\n", lrp.DtParam)
		bs2, _ := json.Marshal(lrp.DtParam)
		var out2 bytes.Buffer
		_ = json.Indent(&out2, bs2, "", "\t")
		logger.Log.Printf("structure DtParam =%v\n", out2.String())

	} else if algName == common.LinearRegressionAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.ModelTraining.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LinearRegressionParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.GBDTAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.ModelTraining.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.GbdtParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

		logger.Log.Printf("[TrainWorker]: original DtParam = %+v\n", lrp.DtParam)
		bs2, _ := json.Marshal(lrp.DtParam)
		var out2 bytes.Buffer
		_ = json.Indent(&out2, bs2, "", "\t")
		logger.Log.Printf("structure DtParam =%v\n", out2.String())

	} else if algName == common.LimeSamplingAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.LimeInsSample.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LimeSamplingParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.LimeCompPredictionAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.LimePred.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LimeCompPredictionParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.LimeCompWeightsAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.LimeWeight.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LimeCompWeightsParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.LimeFeatSelAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.LimeFeature.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LimeFeatSelParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

	} else if algName == common.LimeInterpretAlgName {

		res, _ := base64.StdEncoding.DecodeString(w.DslObj.Tasks.LimeInterpret.InputConfigs.SerializedAlgorithmConfig)
		lrp := v0.LimeInterpretParams{}
		_ = proto.Unmarshal(res, &lrp)

		logger.Log.Printf("[TrainWorker]: original params = %+v\n", lrp)
		bs, _ := json.Marshal(lrp)
		var out bytes.Buffer
		_ = json.Indent(&out, bs, "", "\t")
		logger.Log.Printf("structure params =%v\n", out.String())

		res2, _ := base64.StdEncoding.DecodeString(lrp.InterpretModelParam)
		lrp2 := v0.LinearRegressionParams{}
		_ = proto.Unmarshal(res2, &lrp2)
		logger.Log.Printf("[TrainWorker]: original LinearRegressionParams = %+v\n", lrp2)
		bs2, _ := json.Marshal(lrp2)
		var out2 bytes.Buffer
		_ = json.Indent(&out2, bs2, "", "\t")
		logger.Log.Printf("structure LinearRegressionParams =%v\n", out2.String())

	} else {

	}

	logger.Log.Println("[TrainWorker]: ----------- WorkerAddr , printParams Done-----------")

}
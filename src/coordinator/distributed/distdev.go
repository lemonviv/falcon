package distributed

import (
	"coordinator/cache"
	c "coordinator/client"
	"coordinator/common"
	"coordinator/distributed/worker"
	"coordinator/logger"
)



func SetupDistDev(qItem *cache.QItem, workerType string) {
	/**
	 * @Author
	 * @Description run master, and then, master will call lister to run worker
	 * @Date 2:36 下午 5/12/20
	 * @Param
	 * @return
	 **/

	masterPort := c.GetFreePort(common.CoordinatorUrl)
	logger.Do.Println("SetupDist: Launch master Get port", masterPort)

	masterIp := common.CoordIP
	masterAddress := masterIp + ":" + masterPort

	logger.Do.Println("SetupDist: Launch master DevEnv")

	// use a thread
	SetupMaster(masterAddress, qItem, workerType)

	logger.Do.Println("SetupDist: setup master done")
}


func SetupWorkerHelperDev(masterAddress, workerType, jobId, dataPath, modelPath, dataOutput string)  {

	/**
	 * @Author
	 * @Description: this func is only called by partyserver
	 * @Date 2:14 下午 1/12/20
	 * @Param
	 	httpHost： 		IP of the partyserver address
		masterAddress： IP of the master address
		masterAddress： train or predictor
	 **/
	logger.Do.Println("SetupWorkerHelper: Creating parameters:", masterAddress, workerType)

	workerPort := c.GetFreePort(common.CoordinatorUrl)

	workerAddress := common.PartyServerIP + ":" + workerPort
	var serviceName string

	// in dev, use thread

	common.TaskDataPath = dataPath
	common.TaskDataOutput = dataOutput
	common.TaskModelPath = modelPath

	if workerType == common.TrainWorker{

		serviceName = "worker-jid" + jobId + "-train-" + common.PartyServerId
		common.TaskRuntimeLogs = common.PartyServeBasePath+"/"+"run_time_logs/"+serviceName

		logger.Do.Println("SetupWorkerHelper: Current in Dev, TrainWorker")

		wk := worker.InitTrainWorker(masterAddress, workerAddress)
		wk.RunWorker(wk)

	}else if workerType == common.PredictWorker{

		serviceName = "worker-jid" + jobId + "-predict-" + common.PartyServerId
		common.TaskRuntimeLogs = common.PartyServeBasePath+"/"+"run_time_logs/"+serviceName

		logger.Do.Println("SetupWorkerHelper: Current in Dev, PredictWorker")

		wk := worker.InitPredictWorker(masterAddress, workerAddress)
		wk.RunWorker(wk)

	}

		// in prod, use k8s to run train/predict server as a isolate process

	logger.Do.Println("SetupDist: worker is running")

}
package main

import (
	"coordinator/coordserver"
	"coordinator/cache"
	"coordinator/common"
	"coordinator/distributed"
	"coordinator/distributed/prediction"
	"coordinator/distributed/worker"
	"coordinator/partyserver"
	"coordinator/logger"
	"fmt"
	"os"
	"runtime"
	"time"
)

func init() {

	// prority: env >  user provided > default value
	runtime.GOMAXPROCS(4)
	initLogger()
	InitEnvs(common.ServiceNameGlobal)

}

func initLogger(){
	// this path is fixed, used to creating folder inside container
	var fixedPath string
	if common.Env == common.DevEnv{
		common.LocalPath = os.Getenv("DATA_BASE_PATH")
		fixedPath = common.LocalPath+"runtimeLogs"
	}else{
		fixedPath ="./logs"
	}

	fmt.Println("Loging to ", fixedPath)

	_ = os.Mkdir(fixedPath, os.ModePerm)
	// Use layout string for time format.
	const layout = "2006-01-02T15:04:05"
	// Place now in the string.
	rawTime := time.Now()

	var logFileName string
	logFileName = fixedPath + "/" + common.ServiceNameGlobal + rawTime.Format(layout) + "logs"

	logger.Do, logger.F = logger.GetLogger(logFileName)
}


func getCoordUrl(url string) string{
		// using service name+ port to connect to coord
	    logger.Do.Printf("<<<<<<<<<<<<<<<<< Read envs, User defined,   key: CoordSvcURLGlobal, value: %s >>>>>>>>>>>>>\n",url)
		return url
}


func InitEnvs(svcName string){

	if svcName=="coord"{
		// coord needs db information
		common.JobDbEngine       = common.GetEnv("MS_ENGINE", "sqlite3")
		common.JobDbSqliteDb     = common.GetEnv("MS_SQLITE_DB", "falcon")
		common.JobDbHost         = common.GetEnv("MS_HOST","localhost")
		common.JobDbMysqlUser    = common.GetEnv("MS_MYSQL_USER", "falcon")
		common.JobDbMysqlPwd     = common.GetEnv("MS_MYSQL_PWD", "falcon")
		common.JobDbMysqlDb      = common.GetEnv("MS_MYSQL_DB", "falcon")
		common.JobDbMysqlOptions = common.GetEnv("MS_MYSQL_OPTIONS", "?parseTime=true")
		common.JobDbMysqlPort    = common.GetEnv("MYSQL_CLUSTER_PORT", "30000")

		common.RedisHost      = common.GetEnv("REDIS_HOST","localhost")
		common.RedisPwd       = common.GetEnv("REDIS_PWD", "falcon")
		// coord needs redis information
		common.RedisPort       = common.GetEnv("REDIS_CLUSTER_PORT", "30002")
		// find the cluster port, call internally
		common.JobDbMysqlNodePort    = common.GetEnv("MYSQL_NODE_PORT", "30001")
		common.RedisNodePort    = common.GetEnv("REDIS_NODE_PORT", "30003")

		// find the cluster port, call internally
		common.CoordAddrGlobal = common.GetEnv("COORDINATOR_IP", "")
		common.CoordPort   = common.GetEnv("COORD_TARGET_PORT", "30004")

		common.CoordSvcName = common.GetEnv("COORD_SVC_NAME", "")

		common.CoordSvcURLGlobal = getCoordUrl(common.CoordAddrGlobal + ":" + common.CoordPort)

		if len(common.ServiceNameGlobal) == 0{
			logger.Do.Println("Error: Input Error, ServiceNameGlobal is either 'coord' or 'partyserver' ")
			os.Exit(1)
		}

	}else if svcName=="partyserver"{

		// partyserver needs coord ip+port,lis port
		common.CoordAddrGlobal = common.GetEnv("COORDINATOR_IP", "")
		common.CoordPort = common.GetEnv("COORD_TARGET_PORT", "30004")
		common.ListenAddrGlobal = common.GetEnv("PARTYSERVER_IP", "")
		common.ListenBasePath = common.GetEnv("DATA_BASE_PATH", "")

		// partyserver communicate coord with ip+port
		common.CoordSvcURLGlobal = getCoordUrl(common.CoordAddrGlobal + ":" + common.CoordPort)

		// run partyserver requires to get a new partyserver port
		common.PartyServerPort = common.GetEnv("PARTYSERVER_NODE_PORT", "")

		common.PartyServerId = common.GetEnv("PARTY_NUMBER", "")

		// partyserver needs will send this to coord
		common.ListenURLGlobal = common.ListenAddrGlobal + ":" + common.PartyServerPort

		if common.CoordAddrGlobal=="" || common.ListenAddrGlobal==""||common.PartyServerPort=="" {
			logger.Do.Println("Error: Input Error, either CoordAddrGlobal or ListenAddrGlobal not provided")
			os.Exit(1)
		}

	}else if svcName==common.MasterExecutor {

		// master needs redis information
		common.RedisHost      = common.GetEnv("REDIS_HOST","localhost")
		common.RedisPwd       = common.GetEnv("REDIS_PWD", "falcon")
		common.RedisPort       = common.GetEnv("REDIS_CLUSTER_PORT", "30002")
		common.RedisNodePort    = common.GetEnv("REDIS_NODE_PORT", "30003")
		common.CoordPort = common.GetEnv("COORD_TARGET_PORT", "30004")

		// master needs queue item, task type
		common.MasterQItem =common.GetEnv("ITEM_KEY", "")
		common.ExecutorTypeGlobal = common.GetEnv("EXECUTOR_TYPE", "")
		common.MasterURLGlobal = common.GetEnv("MASTER_URL", "")

		common.CoordSvcName = common.GetEnv("COORD_SVC_NAME", "")

		common.ExecutorCurrentName = common.GetEnv("EXECUTOR_NAME", "")

		// master communicate coord with ip+port in dev, with name+port in prod
		if common.Env==common.DevEnv{

			logger.Do.Println("CoordAddrGlobal: ", common.CoordAddrGlobal  + ":" + common.CoordPort)

			common.CoordSvcURLGlobal = getCoordUrl(common.CoordAddrGlobal + ":" + common.CoordPort)

		}else if common.Env==common.ProdEnv{

			logger.Do.Println("CoordSvcName: ", common.CoordSvcName  + ":" + common.CoordPort)

			common.CoordSvcURLGlobal = getCoordUrl(common.CoordSvcName  + ":" + common.CoordPort)
		}


		if common.CoordSvcURLGlobal==""{
			logger.Do.Println("Error: Input Error, CoordSvcURLGlobal not provided")
			os.Exit(1)
		}

	}else if svcName==common.TrainExecutor{
		// this will be executed only in production, in dev, the common.ExecutorTypeGlobal==""

		common.TaskDataPath = common.GetEnv("TASK_DATA_PATH", "")
		common.TaskModelPath = common.GetEnv("TASK_MODEL_PATH", "")
		common.TaskDataOutput = common.GetEnv("TASK_DATA_OUTPUT", "")
		common.TaskRuntimeLogs = common.GetEnv("RUN_TIME_LOGS", "")

		common.ExecutorTypeGlobal = common.GetEnv("EXECUTOR_TYPE", "")
		common.WorkerURLGlobal = common.GetEnv("WORKER_URL", "")
		common.MasterURLGlobal = common.GetEnv("MASTER_URL", "")
		common.ExecutorCurrentName = common.GetEnv("EXECUTOR_NAME", "")
		if common.MasterURLGlobal=="" || common.WorkerURLGlobal=="" {
			logger.Do.Println("Error: Input Error, either WorkerAddrGlobal or MasterAddrGlobal or TaskTypeGlobal not provided")
			os.Exit(1)
		}

	}else if svcName==common.PredictExecutor{

		common.TaskDataPath = common.GetEnv("TASK_DATA_PATH", "")
		common.TaskModelPath = common.GetEnv("TASK_MODEL_PATH", "")
		common.TaskDataOutput = common.GetEnv("TASK_DATA_OUTPUT", "")
		common.TaskRuntimeLogs = common.GetEnv("RUN_TIME_LOGS", "")

		// this will be executed only in production, in dev, the common.ExecutorTypeGlobal==""

		common.ExecutorTypeGlobal = common.GetEnv("EXECUTOR_TYPE", "")
		common.WorkerURLGlobal = common.GetEnv("WORKER_URL", "")
		common.MasterURLGlobal = common.GetEnv("MASTER_URL", "")
		common.ExecutorCurrentName = common.GetEnv("EXECUTOR_NAME", "")
		if common.MasterURLGlobal=="" || common.WorkerURLGlobal=="" {
			logger.Do.Println("Error: Input Error, either WorkerAddrGlobal or MasterAddrGlobal or TaskTypeGlobal not provided")
			os.Exit(1)
		}
	}
}


func main() {

	defer logger.HandleErrors()

	defer func(){
		_=logger.F.Close()
	}()


	if common.ServiceNameGlobal == "coord" {
		logger.Do.Println("Launch coordinator_server, the common.ServiceNameGlobal", common.ServiceNameGlobal)

		coordserver.SetupHttp(3)
	}

	// start work in remote machine automatically
	if common.ServiceNameGlobal == "partyserver" {

		logger.Do.Println("Launch coordinator_server, the common.ServiceNameGlobal", common.ServiceNameGlobal)

		partyserver.SetupPartyServer()
	}


	//////////////////////////////////////////////////////////////////////////
	//						 start tasks, called internally 				//
	// 																	    //
	//////////////////////////////////////////////////////////////////////////

	//those 2 is only called internally

	if common.ServiceNameGlobal == common.MasterExecutor {

		logger.Do.Println("Lunching coordinator_server, the common.ExecutorTypeGlobal", common.ExecutorTypeGlobal)

		// this should be the service name, defined at runtime,
		masterUrl := common.MasterURLGlobal

		qItem := cache.Deserialize(cache.InitRedisClient().Get(common.MasterQItem))

		taskType := common.ExecutorTypeGlobal

		distributed.SetupMaster(masterUrl, qItem, taskType)

	}

	if common.ServiceNameGlobal == common.TrainExecutor {

		logger.Do.Println("Lunching coordinator_server, the common.ExecutorTypeGlobal", common.ExecutorTypeGlobal)

		worker.RunWorker(common.MasterURLGlobal, common.WorkerURLGlobal)
	}

	if common.ServiceNameGlobal == common.PredictExecutor {

		logger.Do.Println("Lunching coordinator_server, the common.ExecutorTypeGlobal", common.ExecutorTypeGlobal)

		prediction.RunPrediction(common.MasterURLGlobal, common.WorkerURLGlobal)

	}


}

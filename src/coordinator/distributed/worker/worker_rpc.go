package worker

import (
	"coordinator/config"
	"coordinator/distributed/taskmanager"
	"log"
	"net/rpc"
	"sync"
)

func RunWorker(masterAddress, workerProxy, workerHost, workerPort string, wg *sync.WaitGroup) {
	ServiceName := "Worker"
	workerAddress := workerHost + ":" + workerPort

	wk := new(Worker)
	wk.InitRpc(workerProxy, workerAddress)

	wk.SuicideTimeout = config.WorkerTimeout

	// the lock needs to pass to multi funcs, must create a instance
	wk.pm = taskmanager.InitSubProcessManager()
	wk.taskFinish = make(chan bool)

	wk.reset()
	go wk.eventLoop()


	rpcSvc := rpc.NewServer()
	err := rpcSvc.Register(wk)
	if err!= nil{
		log.Printf("%s: start Error \n", ServiceName)
		return
	}

	log.Println("Worker: register to masterAddress= ", masterAddress)
	wk.register(masterAddress)

	wk.StartRPCServer(rpcSvc, ServiceName, true)
	wg.Done()

	log.Println("Worker: ", workerAddress, "runWorker exit")
}

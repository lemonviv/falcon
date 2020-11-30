package distributed

import (
	c "coordinator/client"
	"coordinator/config"
	"coordinator/distributed/master"
	"coordinator/distributed/prediction"
	"coordinator/distributed/taskmanager"
	"coordinator/distributed/utils"
	"coordinator/distributed/worker"
	"fmt"
	"log"
	"strconv"
	"sync"
)

func SetupDist(httpHost, httpPort string, qItem *config.QItem, taskType string) {
	log.Println("SetupDist: Lunching master")

	httpAddr := httpHost + ":" + httpPort
	port, e := utils.GetFreePort()

	if e != nil {
		log.Println("Get port Error")
		return
	}
	masterAddress := httpHost + ":" + fmt.Sprintf("%d", port)
	ms := master.RunMaster("tcp", masterAddress, httpAddr, qItem)

	// update job's master address
	c.JobUpdateMaster(httpAddr, masterAddress, qItem.JobId)

	for _, ip := range qItem.IPs {

		// lunching the worker

		// todo currently worker port is fixed, not a good design, change to dynamic later
		// maybe check table wit ip, and + port got from table also

		// send a request to http
		c.SetupWorker(ip+":"+config.ListenerPort, masterAddress, taskType)
	}

	// wait until job done
	ms.Wait()
}

func SetupWorker(httpHost string, masterAddress string) error {
	log.Println("SetupDist: Lunching worker threads")

	wg := sync.WaitGroup{}

	// each listener only have 1 worker thread

	for i := 0; i < 1; i++ {
		port, e := utils.GetFreePort()
		if e != nil {
			log.Println("SetupDist: Lunching worker Get port Error")
			return e
		}
		wg.Add(1)

		// worker address share the same host with listener server
		go worker.RunWorker(masterAddress, "tcp", httpHost, fmt.Sprintf("%d", port), &wg)
	}
	wg.Wait()
	return nil
}

func KillJob(masterAddr, Proxy string) {
	ok := c.Call(masterAddr, Proxy, "Master.KillJob", new(struct{}), new(struct{}))
	if ok == false {
		log.Println("Master: KillJob error")
		panic("Master: KillJob error")
	} else {
		log.Println("Master: KillJob Done")
	}
}

func SetupPredictionHelper(httpHost string, masterAddress string) error {

	dir:=""
	stdIn := "input from keyboard"
	commend := ""
	args := []string{"/coordinator_server", "-svc predictor -b 1"}
	var envs []string

	pm := taskmanager.InitSubProcessManager()
	pm.IsWait = false

	killed, e, el, ol := pm.ExecuteSubProc(dir, stdIn, commend, args, envs)
	log.Println(killed, e, el, ol)

	return nil
}


func SetupPrediction(httpHost string) {
	log.Println("SetupDist: Lunching prediction svc")

	port, e := utils.GetFreePort()
	if e != nil {
		log.Println("SetupDist: Lunching worker Get port Error")
	}

	sPort := strconv.Itoa(port)

	prediction.RunPrediction(httpHost, sPort,"tcp")

}

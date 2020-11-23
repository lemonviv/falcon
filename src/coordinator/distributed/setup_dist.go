package distributed

import (
	c "coordinator/client"
	"coordinator/config"
	"coordinator/distributed/master"
	"coordinator/distributed/utils"
	"coordinator/distributed/worker"
	"fmt"
	"log"
	"sync"
)

func SetupDist(httpHost, httpPort string, qItem *config.QItem) {
	log.Println("SetupDist: Launch master")

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

		// Launch the worker

		// todo currently worker port is fixed, not a good design, change to dynamic later
		// maybe check table wit ip, and + port got from table also

		// send a request to http
		c.SetupWorker(ip+":"+config.ListenerPort, masterAddress)
	}

	// wait until job done
	ms.Wait()
}

func SetupWorker(httpHost string, masterAddress string) error {
	log.Println("SetupDist: Launch worker threads")

	wg := sync.WaitGroup{}

	// each listener only have 1 worker thread

	for i := 0; i < 1; i++ {
		port, e := utils.GetFreePort()
		if e != nil {
			log.Println("SetupDist: Launch worker Get port Error")
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
	ok := utils.Call(masterAddr, Proxy, "Master.KillJob", new(struct{}), new(struct{}))
	if ok == false {
		log.Println("Master: KillJob error")
		panic("Master: KillJob error")
	} else {
		log.Println("Master: KillJob Done")
	}
}

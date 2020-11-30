package main

import (
	"coordinator/api"
	"coordinator/config"
	"coordinator/distributed"
	"coordinator/listener"
	"flag"
	"log"
	"os"
	"runtime"
	"strings"
)

var svc string
var httpAddr string
var listenerAddr string
var predictorAddr string

func init() {
	runtime.GOMAXPROCS(4)
	flag.StringVar(&svc, "svc", "coordinator", "choose which service to run, 'coordinator' or 'listener'")
	flag.StringVar(&httpAddr, "cip", "", "Ip Address of coordinator")
	flag.StringVar(&listenerAddr, "lip", "", "Ip Address of listener")
	flag.StringVar(&predictorAddr, "pip", "", "Ip Address of predictor")
}

func verifyArgs() {
	if len(httpAddr) == 0 {
		log.Println("Error: Input Error, Must Provide ip of coordinator")
		os.Exit(1)
	}

	if !(strings.Contains(svc, "coordinator") || strings.Contains(svc, "listener")) {
		log.Println("Error: Input Error, svc is either 'coordinator' or 'listener'")
		os.Exit(1)
	}
}

func main() {
	flag.Parse()
	verifyArgs()

	_ = os.Mkdir(".logs", os.ModePerm)
	logFileName := ".logs/" + svc + ".log"

	logFile, logErr := os.OpenFile(logFileName, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0666)

	if logErr != nil {
		log.Println("Fail to find", logFile, "cServer start Failed")
		os.Exit(1)
	}

	log.SetOutput(logFile)

	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	// start work in remote machine automatically
	if svc == "listener" {

		if len(listenerAddr) == 0 {
			log.Println("Error: Input Error, Must Provide ip of listener")
			os.Exit(1)
		}
		log.Println("Lunching coordinator_server, the svc", svc)

		masterAddr := httpAddr + ":" + config.MasterPort
		listener.SetupListener(listenerAddr, config.ListenerPort, masterAddr)
	}

	if svc == "coordinator" {
		log.Println("Lunching coordinator_server, the svc", svc)

		api.SetupHttp(httpAddr, config.MasterPort, 3)
	}

	if svc == "predictor" {

		if len(predictorAddr) == 0 {
			log.Println("Error: Input Error, Must Provide ip of predictor ")
			os.Exit(1)
		}
		log.Println("Lunching coordinator_server, the svc", svc)

		distributed.SetupPrediction(predictorAddr)

	}
}

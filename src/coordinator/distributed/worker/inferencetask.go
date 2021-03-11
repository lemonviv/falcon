package worker

import (
	"coordinator/common"
	"coordinator/distributed/entity"
	"coordinator/logger"
	"os/exec"
)

func (wk *InferenceWorker) CreateInference(doTaskArgs *entity.DoTaskArgs) {
	// todo gobuild.sh sub process to run prediction job

	logger.Do.Println("InferenceWorker: CreateService")

	cmd := exec.Command("python3", "/go/preprocessing.py", "-a=1", "-b=2")

	var envs []string

	// 2 thread will ready from isStop channel, only one is running at the any time

	el, e := wk.Pm.CreateResources(cmd, envs)

	logger.Do.Println("Worker:task 1 pre processing done", el)

	if e != common.SubProcessNormal {
		// return res is used to control the rpc call status, always return nil, but
		// keep error at rep.Errs
		return
	}

}

func (wk *InferenceWorker) UpdateInference() error {
	return nil
}

func (wk *InferenceWorker) QueryInference() error {
	return nil
}
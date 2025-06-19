import math
from PowerStruc import power_struc_carbon
from greenPower import carbon_intensity

class Machine:
    def __init__(self, id):
        self.id = id
        self.running_job_id = -1
        self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id):
        if self.is_free:
            self.running_job_id = job_id
            self.is_free = False
            self.job_history.append(job_id)
            return True
        else:
            return False

    def release(self):
        if self.is_free:
            return -1
        else:
            self.is_free = True
            self.running_job_id = -1
            return 1

    def reset(self):
        self.is_free = True
        self.running_job_id = -1
        self.job_history = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "


class Cluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node, processor_per_machine, idlePower, greenWin, year=2021):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []
        self.statPower = idlePower*math.ceil(self.total_node / processor_per_machine)
        self.PowerStruc = power_struc_carbon(self.statPower)
        self.carbonIntensity = carbon_intensity(greenWin, year)
        self.green_win = greenWin
        
        # Verification: Ensure power_struc_carbon is properly initialized
        assert hasattr(self.PowerStruc, 'jobPowerLogs'), "PowerStruc should have jobPowerLogs for per-job tracking"
        assert hasattr(self.PowerStruc, 'update'), "PowerStruc should have update method"
        assert hasattr(self.PowerStruc, 'getJobPowerConsumption'), "PowerStruc should have getJobPowerConsumption method"

        for i in range(self.total_node):
            self.all_nodes.append(Machine(i))

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes > self.free_node:
            return False
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes <= self.free_node:
            return True

        request_node = int(math.ceil(float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = []
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = 0

        for m in self.all_nodes:
            if allocated == request_node:
                return allocated_nodes
            if m.taken_by_job(job_id):
                allocated += 1
                self.used_node += 1
                self.free_node -= 1
                allocated_nodes.append(m)

        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)

        for m in releases:
            m.release()

    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node
        self.PowerStruc = power_struc_carbon(self.statPower)
        for m in self.all_nodes:
            m.reset()

    def backfill_check(self,running_jobs,job,current_time,backfill=1):
        if not self.can_allocated(job):
            return False
        if backfill==2:
            return True
        currentSlot=self.PowerStruc.getSlotFromRunning(running_jobs, current_time)
        slotList,beforeList=self.PowerStruc.getPre(current_time,
                                                        current_time + job.request_time,
                                                        job.power,currentSlot)
        totalEnergyBefore=0
        carbonEmissionsBefore=0
        totalEnergyAfter=0
        carbonEmissionsAfter=0
        minIndex=int(current_time/3600)
        maxIndex=minIndex+self.green_win-1

        for i in range(len(slotList) - 1):
            powerAfter = slotList[i]['power']
            startAfter = slotList[i]['timeSlot']
            endAfter = slotList[i + 1]['timeSlot']
            lastTimeAfter = endAfter - startAfter
            consumeEnergyAfter = lastTimeAfter * powerAfter
            totalEnergyAfter += consumeEnergyAfter
            inc_carbonEmissionsAfter = self.carbonIntensity.getCarbonEmissionsEstimate(powerAfter, startAfter, endAfter,minIndex,maxIndex)
            carbonEmissionsAfter += inc_carbonEmissionsAfter

            powerBefore = beforeList[i]['power']
            startBefore = beforeList[i]['timeSlot']
            endBefore = beforeList[i + 1]['timeSlot']
            lastTimeBefore = endBefore - startBefore
            consumeEnergyBefore = lastTimeBefore * powerBefore
            totalEnergyBefore += consumeEnergyBefore
            inc_carbonEmissionsBefore = self.carbonIntensity.getCarbonEmissionsEstimate(powerBefore, startBefore, endBefore,minIndex,maxIndex)
            carbonEmissionsBefore += inc_carbonEmissionsBefore

        jobCarbonEmissions = carbonEmissionsAfter - carbonEmissionsBefore

        # return True #EASY

        if jobCarbonEmissions < 50000:  # Threshold in gCO2eq - may need adjustment
            return True
        else:
            return False

    def LPTPN_check(self,running_jobs,job,current_time):

        currentSlot=self.PowerStruc.getSlotFromRunning(running_jobs, current_time)
        slotList,beforeList=self.PowerStruc.getPre(current_time,
                                                        current_time + job.request_time,
                                                        job.power,currentSlot)
        totalEnergyBefore=0
        carbonEmissionsBefore=0
        totalEnergyAfter=0
        carbonEmissionsAfter=0
        minIndex=int(current_time/3600)
        maxIndex=minIndex+self.green_win-1

        for i in range(len(slotList) - 1):
            powerAfter = slotList[i]['power']
            startAfter = slotList[i]['timeSlot']
            endAfter = slotList[i + 1]['timeSlot']
            lastTimeAfter = endAfter - startAfter
            consumeEnergyAfter = lastTimeAfter * powerAfter
            totalEnergyAfter += consumeEnergyAfter
            inc_carbonEmissionsAfter = self.carbonIntensity.getCarbonEmissionsEstimate(powerAfter, startAfter, endAfter,minIndex,maxIndex)
            carbonEmissionsAfter += inc_carbonEmissionsAfter

            powerBefore = beforeList[i]['power']
            startBefore = beforeList[i]['timeSlot']
            endBefore = beforeList[i + 1]['timeSlot']
            lastTimeBefore = endBefore - startBefore
            consumeEnergyBefore = lastTimeBefore * powerBefore
            totalEnergyBefore += consumeEnergyBefore
            inc_carbonEmissionsBefore = self.carbonIntensity.getCarbonEmissionsEstimate(powerBefore, startBefore, endBefore,minIndex,maxIndex)
            carbonEmissionsBefore += inc_carbonEmissionsBefore

        jobCarbonEmissions = carbonEmissionsAfter - carbonEmissionsBefore

        if jobCarbonEmissions < 50000:  # Threshold in gCO2eq - may need adjustment
            return True
        else:
            return False

    def backfill_check_ga(self,running_jobs,job,current_time,minGreen,backfill=1):
        if not self.can_allocated(job):
            return False
        if backfill==2:
            return True
        currentSlot=self.PowerStruc.getSlotFromRunning(running_jobs, current_time)
        slotList,beforeList=self.PowerStruc.getPre(current_time,
                                                        current_time + job.request_time,
                                                        job.power,currentSlot)
        totalEnergyBefore=0
        carbonEmissionsBefore=0
        totalEnergyAfter=0
        carbonEmissionsAfter=0
        minIndex=int(minGreen/3600)
        maxIndex=minIndex+self.green_win-1
        for i in range(len(slotList) - 1):
            powerAfter = slotList[i]['power']
            startAfter = slotList[i]['timeSlot']
            endAfter = slotList[i + 1]['timeSlot']
            lastTimeAfter = endAfter - startAfter
            consumeEnergyAfter = lastTimeAfter * powerAfter
            totalEnergyAfter += consumeEnergyAfter
            inc_carbonEmissionsAfter = self.carbonIntensity.getCarbonEmissionsEstimate(powerAfter, startAfter, endAfter,minIndex,maxIndex)
            carbonEmissionsAfter += inc_carbonEmissionsAfter

            powerBefore = beforeList[i]['power']
            startBefore = beforeList[i]['timeSlot']
            endBefore = beforeList[i + 1]['timeSlot']
            lastTimeBefore = endBefore - startBefore
            consumeEnergyBefore = lastTimeBefore * powerBefore
            totalEnergyBefore += consumeEnergyBefore
            inc_carbonEmissionsBefore = self.carbonIntensity.getCarbonEmissionsEstimate(powerBefore, startBefore, endBefore,minIndex,maxIndex)
            carbonEmissionsBefore += inc_carbonEmissionsBefore

        jobCarbonEmissions = carbonEmissionsAfter - carbonEmissionsBefore

        if jobCarbonEmissions < 50000:  # Threshold in gCO2eq - may need adjustment
            return True
        else:
            return False


class FakeList:
    def __init__(self, l):
        self.len = l
    def __len__(self):
        return self.len

class SimpleCluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1:
            if job.request_number_of_nodes > self.free_node:
                return False
            else:
                return True

        request_node = int(math.ceil(float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = FakeList(0)
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = request_node

        self.used_node += allocated
        self.free_node -= allocated
        allocated_nodes.len = allocated
        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)


    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node


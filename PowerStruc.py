class power_struc():
    def __init__(self, statPower):
        self.powerSlotLog=[]
        self.currentIndex=0
        self.statPower = statPower


    def reset(self):
        self.powerSlotLog=[]
        self.currentIndex=0

    def update(self,start,end,power):

        head_index = 0
        if len(self.powerSlotLog)==0:
            self.powerSlotLog.append({'timeSlot': start, 'power': power+self.statPower})
            self.powerSlotLog.append({'timeSlot': end, 'power': self.statPower})
            return
        for i in range(self.currentIndex,len(self.powerSlotLog)):
            if start>self.powerSlotLog[i]['timeSlot']:
                if i==len(self.powerSlotLog)-1:
                    self.powerSlotLog.append({'timeSlot': start, 'power': power+self.statPower})
                    self.powerSlotLog.append({'timeSlot': end, 'power': self.statPower})
                    return
                continue
            elif start==self.powerSlotLog[i]['timeSlot']:
                head_index=i
                self.powerSlotLog[i]['power'] += power
                if head_index == len(self.powerSlotLog) - 1:
                    self.powerSlotLog.append({'timeSlot': end, 'power': self.powerSlotLog[head_index]['power'] - power})
                    return
                break
            else:
                head_index=i
                beforeIndex=i-1
                newSloat={'timeSlot': start, 'power': power+self.powerSlotLog[beforeIndex]['power']}
                self.powerSlotLog.insert(head_index,newSloat)
                break

        for i in range(head_index+1,len(self.powerSlotLog)):
            if end>self.powerSlotLog[i]['timeSlot']:
                self.powerSlotLog[i]['power'] += power
                if i==len(self.powerSlotLog)-1:
                    self.powerSlotLog.append({'timeSlot': end, 'power': self.powerSlotLog[i]['power'] - power})
                    return
                continue
            elif end==self.powerSlotLog[i]['timeSlot']:
                return
            else:
                beforeIndex=i-1
                newSloat={'timeSlot': end, 'power': self.powerSlotLog[beforeIndex]['power']-power}
                self.powerSlotLog.insert(i,newSloat)
                return

    def updateCurrentTime(self,updateTime):
        for i in range(self.currentIndex,len(self.powerSlotLog)):
            if self.powerSlotLog[i]['timeSlot']==updateTime:
                self.currentIndex=i
                break
            if self.powerSlotLog[i]['timeSlot']>updateTime:
                if i>0:
                    self.currentIndex=i-1
                break
            if i==len(self.powerSlotLog)-1:
                self.currentIndex=len(self.powerSlotLog)-1
        return


    def getSlotFromRunning(self,running_jobs,currentTime):
        running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        currentSlot=[]
        if len(running_jobs)==0:
            return currentSlot
        lastPower=self.statPower
        lastJobPower =0
        for job in reversed(running_jobs):
            end=job.scheduled_time + job.request_time
            power=lastPower+lastJobPower
            lastPower=power
            currentSlot.append({'timeSlot': end, 'power': power})
            lastJobPower=job.power
        currentSlot.append({'timeSlot': currentTime, 'power': lastPower+lastJobPower})
        return currentSlot[::-1]

    def getPre(self,start,end,power,currentSlot):
        slotList=[]
        beforeList=[]
        head_index = 0
        if len(currentSlot)==0:
            slotList.append({'timeSlot': start, 'power': power+self.statPower})
            slotList.append({'timeSlot': end, 'power': self.statPower})
            beforeList.append({'timeSlot': start, 'power': self.statPower})
            beforeList.append({'timeSlot': end, 'power': self.statPower})
            return slotList,beforeList
        for i in range(len(currentSlot)):
            if start>currentSlot[i]['timeSlot']:
                if i==len(currentSlot)-1:
                    slotList.append({'timeSlot': start, 'power': power+self.statPower})
                    slotList.append({'timeSlot': end, 'power': self.statPower})
                    beforeList.append({'timeSlot': start, 'power': self.statPower})
                    beforeList.append({'timeSlot': end, 'power': self.statPower})
                    return slotList, beforeList
                continue
            elif start==currentSlot[i]['timeSlot']:
                head_index=i
                slotList.append({'timeSlot': start, 'power': currentSlot[i]['power'] + power})
                beforeList.append({'timeSlot': start, 'power': currentSlot[i]['power']})
                if head_index == len(currentSlot) - 1:
                    slotList.append({'timeSlot': end, 'power': currentSlot[head_index]['power']})
                    beforeList.append({'timeSlot': end, 'power': currentSlot[head_index]['power']})
                    return slotList,beforeList
                break
            else:
                head_index=i
                beforeIndex=i-1
                newSloat={'timeSlot': start, 'power': power+currentSlot[beforeIndex]['power']}
                slotList.append(newSloat)
                beforeList.append({'timeSlot': start, 'power': currentSlot[beforeIndex]['power']})
                break

        for i in range(head_index,len(currentSlot)):
            if end>currentSlot[i]['timeSlot']:
                slotList.append({'timeSlot': currentSlot[i]['timeSlot'], 'power':currentSlot[i]['power'] + power})
                beforeList.append({'timeSlot': currentSlot[i]['timeSlot'], 'power':currentSlot[i]['power']})
                if i==len(currentSlot)-1:
                    slotList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                    beforeList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                    return slotList,beforeList
                continue
            elif end==currentSlot[i]['timeSlot']:
                slotList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                beforeList.append({'timeSlot': end, 'power': currentSlot[i]['power']})
                return slotList,beforeList
            else:
                beforeIndex=i-1
                newSloat={'timeSlot': end, 'power': currentSlot[beforeIndex]['power']}
                slotList.append(newSloat)
                beforeList.append({'timeSlot': end, 'power': currentSlot[beforeIndex]['power']})
                return slotList,beforeList


class power_struc_carbon():
    """
    Carbon-aware power structure that tracks power consumption per job
    and maintains separate idle power tracking for carbon footprint analysis.
    """
    
    def __init__(self, statPower):
        # Job-specific power tracking: {job_id: [time_slots]}
        self.jobPowerLogs = {}  # job_id -> list of {'timeSlot': time, 'power': watts}
        
        # Idle power tracking: constant baseline power when no jobs are running
        self.idlePowerLog = []  # list of {'timeSlot': time, 'power': idle_watts}
        
        # Current state
        self.currentIndex = 0
        self.statPower = statPower  # idle/baseline power in watts
        self.currentTime = 0
        
        # Initialize with idle power
        self.idlePowerLog.append({'timeSlot': 0, 'power': self.statPower})

    def reset(self):
        """Reset all power tracking for a new episode"""
        self.jobPowerLogs = {}
        self.idlePowerLog = []
        self.currentIndex = 0
        self.currentTime = 0
        # Initialize with idle power
        self.idlePowerLog.append({'timeSlot': 0, 'power': self.statPower})

    def update(self, start, end, power, job_id):
        """
        Track power consumption for a specific job
        
        Args:
            start: job start time (seconds)
            end: job end time (seconds) 
            power: job power consumption (watts)
            job_id: unique job identifier
        """
        if job_id not in self.jobPowerLogs:
            self.jobPowerLogs[job_id] = []
        
        # Add job power consumption slots
        self.jobPowerLogs[job_id].append({'timeSlot': start, 'power': power})
        self.jobPowerLogs[job_id].append({'timeSlot': end, 'power': 0})
        
        # Sort to maintain chronological order
        self.jobPowerLogs[job_id].sort(key=lambda x: x['timeSlot'])

    def updateCurrentTime(self, updateTime):
        """Update current time pointer"""
        self.currentTime = updateTime

    def getJobPowerConsumption(self, job_id, start_time=None, end_time=None):
        """
        Get power consumption for a specific job within optional time range
        
        Returns:
            List of power slots for the job
        """
        if job_id not in self.jobPowerLogs:
            return []
        
        slots = self.jobPowerLogs[job_id]
        
        if start_time is None and end_time is None:
            return slots
        
        # Filter by time range if specified
        filtered_slots = []
        for slot in slots:
            if start_time is not None and slot['timeSlot'] < start_time:
                continue
            if end_time is not None and slot['timeSlot'] > end_time:
                continue
            filtered_slots.append(slot)
        
        return filtered_slots

    def getAllJobsPowerAtTime(self, timestamp):
        """
        Get total power consumption from all jobs at a specific timestamp
        
        Returns:
            Dictionary with job_id -> power_watts mapping
        """
        job_powers = {}
        
        for job_id, slots in self.jobPowerLogs.items():
            current_power = 0
            
            # Find the power level at the given timestamp
            for i in range(len(slots)):
                if slots[i]['timeSlot'] <= timestamp:
                    current_power = slots[i]['power']
                else:
                    break
            
            if current_power > 0:
                job_powers[job_id] = current_power
        
        return job_powers

    def getTotalPowerAtTime(self, timestamp):
        """
        Get total cluster power consumption at a specific timestamp
        Includes both job power and idle power
        
        Returns:
            Total power in watts
        """
        job_powers = self.getAllJobsPowerAtTime(timestamp)
        total_job_power = sum(job_powers.values())
        return total_job_power + self.statPower

    def getCarbonTrackingData(self, start_time, end_time):
        """
        Get power consumption data formatted for carbon tracking
        
        Returns:
            Dictionary with:
            - 'jobs': {job_id: [power_slots]}
            - 'idle': [idle_power_slots] 
            - 'total': [total_power_slots]
        """
        # Get all unique timestamps in the range
        timestamps = set()
        timestamps.add(start_time)
        timestamps.add(end_time)
        
        for job_id, slots in self.jobPowerLogs.items():
            for slot in slots:
                if start_time <= slot['timeSlot'] <= end_time:
                    timestamps.add(slot['timeSlot'])
        
        timestamps = sorted(timestamps)
        
        # Build power profiles
        job_profiles = {}
        total_profile = []
        idle_profile = []
        
        for i in range(len(timestamps) - 1):
            current_time = timestamps[i]
            next_time = timestamps[i + 1]
            
            # Get job powers at current time
            job_powers = self.getAllJobsPowerAtTime(current_time)
            total_job_power = sum(job_powers.values())
            
            # Store individual job profiles
            for job_id, power in job_powers.items():
                if job_id not in job_profiles:
                    job_profiles[job_id] = []
                job_profiles[job_id].append({
                    'timeSlot': current_time,
                    'power': power,
                    'duration': next_time - current_time
                })
            
            # Store idle power profile
            idle_profile.append({
                'timeSlot': current_time,
                'power': self.statPower,
                'duration': next_time - current_time
            })
            
            # Store total power profile
            total_profile.append({
                'timeSlot': current_time,
                'power': total_job_power + self.statPower,
                'duration': next_time - current_time
            })
        
        return {
            'jobs': job_profiles,
            'idle': idle_profile,
            'total': total_profile
        }

    def getSlotFromRunning(self, running_jobs, currentTime):
        """
        Backward compatibility method that returns aggregated power slots
        similar to the original power_struc
        """
        if not running_jobs:
            return []
        
        # Create timeline based on job end times
        end_times = []
        for job in running_jobs:
            end_times.append(job.scheduled_time + job.request_time)
        
        end_times = sorted(set(end_times))
        if currentTime not in end_times:
            end_times.append(currentTime)
        end_times.sort()
        
        # Build aggregated slots
        slots = []
        for end_time in end_times:
            total_power = self.getTotalPowerAtTime(end_time)
            slots.append({'timeSlot': end_time, 'power': total_power})
        
        return slots

    def getPre(self, start, end, power, currentSlot):
        """
        Backward compatibility method for backfill calculations
        Returns before and after power profiles for a hypothetical job
        """
        # This is a simplified version - in practice you might want to
        # extend this to work with the new per-job tracking
        slotList = []
        beforeList = []
        
        # For now, use similar logic to original but with job-aware tracking
        if not currentSlot:
            slotList.append({'timeSlot': start, 'power': power + self.statPower})
            slotList.append({'timeSlot': end, 'power': self.statPower})
            beforeList.append({'timeSlot': start, 'power': self.statPower})
            beforeList.append({'timeSlot': end, 'power': self.statPower})
            return slotList, beforeList
        
        # Simple implementation - can be enhanced for more sophisticated carbon tracking
        for slot in currentSlot:
            beforeList.append({'timeSlot': slot['timeSlot'], 'power': slot['power']})
            slotList.append({'timeSlot': slot['timeSlot'], 'power': slot['power'] + power})
        
        return slotList, beforeList


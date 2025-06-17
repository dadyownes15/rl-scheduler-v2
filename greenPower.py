import csv
import os

# Define MAX_CARBON_INTENSITY constant
MAX_CARBON_INTENSITY = 467.25  # Maximum carbon intensity in gCO₂eq/kWh

def dataLoad(fileName, year=2002):
    column = year - 2006
    listByYear = []
    for col in range(column ,column+3):
        f = open(fileName, 'rt')
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            listByYear.append(float(row[col]))

        f.close()
    return listByYear

def wtProducedPower(windSpeedList, turbinePowerNominal,cutInSpeed=2.5,
                    cutOffSpeed=30, ratedOutputSpeed=15):
    wPower = [0.0] * len(windSpeedList)
    for i in range(len(windSpeedList)):
        if windSpeedList[i] > cutInSpeed and windSpeedList[i] < cutOffSpeed:
            if windSpeedList[i] < ratedOutputSpeed:
                wPower[i] = round((turbinePowerNominal *
                                   ((windSpeedList[i]) - (cutInSpeed))) /
                                  ((ratedOutputSpeed) - (cutInSpeed)), 2)
            else:
                wPower[i] = round(turbinePowerNominal, 2)
    return wPower

def solarProducedPower(irradiance, pvEfficiency, numberPv):
    solarPower = [0] * len(irradiance)
    for i in range(len(irradiance)):
        solarPower[i] = round(
            ((pvEfficiency * irradiance[i] * numberPv)), 2)
    return solarPower

def renewablePowerProduced(numberPv,turbinePowerNominal):
    current_dir = os.getcwd()
    wind_file = os.path.join(current_dir, "./data/windspeedS.csv")
    solar_file=os.path.join(current_dir, "./data/irradianceS.csv")

    windListes = dataLoad(wind_file, 2007)
    windPowerLists = wtProducedPower(windListes, turbinePowerNominal)


    pvEfficiency = 0.2
    solarLists = dataLoad(solar_file, 2007)
    solarPowerLists = solarProducedPower(
        solarLists, pvEfficiency, numberPv)

    renewable = [a + b for a, b in zip(solarPowerLists , windPowerLists)]

    return renewable

class carbon_intensity():
    def __init__(self, greenWin, year=2021):
        """
        Initialize carbon intensity class
        greenWin: time window for carbon intensity data
        year: which year column to use from the CSV (2021, 2022, 2023, 2024)
        """
        self.greenWin = greenWin
        self.year = year
        self.carbonIntensityList = self.loadCarbonIntensityData()
    
    def loadCarbonIntensityData(self):
        """Load carbon intensity data from CSV file"""
        current_dir = os.getcwd()
        carbon_file = os.path.join(current_dir, "./data/DK-DK2_hourly_carbon_intensity_noFeb29.csv")
        
        # Map year to column index
        year_to_col = {2021: 1, 2022: 2, 2023: 3, 2024: 4}
        col_index = year_to_col.get(self.year, 1)  # Default to 2021
        
        carbon_list = []
        with open(carbon_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                carbon_list.append(float(row[col_index]))
        
        return carbon_list
    
    def getCarbonIntensitySlot(self, currentTime):
        """
        Get carbon intensity slots for the next greenWin hours
        Returns list of dicts with 'lastTime' and 'carbonIntensity' (gCO2eq/kWh)
        """
        carbonSlot = []
        index = int(currentTime / 3600)  # Current hour index
        t = currentTime
        
        for i in range(index, index + self.greenWin):
            # Handle wrap-around for year-long data
            hour_index = i % len(self.carbonIntensityList)
            carbonIntensity = self.carbonIntensityList[hour_index]
            lastTime = (i + 1) * 3600 - t
            carbonSlot.append({'lastTime': lastTime, 'carbonIntensity': carbonIntensity})
            t = (i + 1) * 3600
        
        return carbonSlot
    
    def getCarbonEmissions(self, power, start, end):
        """
        Calculate total carbon emissions for a given power consumption over time period
        power: power consumption in watts
        start, end: time period in seconds
        Returns: total carbon emissions in gCO2eq
        """
        totalEmissions = 0
        startIndex = int(start / 3600)
        endIndex = int(end / 3600)
        t = start
        
        for i in range(startIndex, endIndex + 1):
            if i == endIndex:
                lastTime = end - t
            else:
                lastTime = (i + 1) * 3600 - t
            
            # Handle wrap-around for year-long data
            hour_index = i % len(self.carbonIntensityList)
            carbonIntensity = self.carbonIntensityList[hour_index]  # gCO2eq/kWh
            
            # Convert power from watts to kW and time from seconds to hours
            energyKWh = (power / 1000.0) * (lastTime / 3600.0)
            emissions = energyKWh * carbonIntensity  # gCO2eq
            totalEmissions += emissions
            
            t = (i + 1) * 3600
        
        return totalEmissions
    
    def getCarbonIntensityUtilization(self, powerSlot):
        """
        Calculate average carbon intensity for the power consumption profile
        Returns: weighted average carbon intensity (gCO2eq/kWh)
        """
        totalEnergy = 0
        totalEmissions = 0
        
        for i in range(len(powerSlot) - 1):
            power = powerSlot[i]['power']
            start = powerSlot[i]['timeSlot']
            end = powerSlot[i + 1]['timeSlot']
            lastTime = end - start
            
            energyKWh = (power / 1000.0) * (lastTime / 3600.0)
            totalEnergy += energyKWh
            
            emissions = self.getCarbonEmissions(power, start, end)
            totalEmissions += emissions
        
        if totalEnergy == 0:
            return 0
        else:
            return totalEmissions / totalEnergy  # Average carbon intensity
    
    def getCarbonEmissionsEstimate(self, power, start, end, minIndex, maxIndex):
        """
        Estimate carbon emissions within a time window (for backfill checks)
        """
        totalEmissions = 0
        startIndex = int(start / 3600)
        endIndex = int(end / 3600)
        t = start
        
        for i in range(startIndex, endIndex + 1):
            if i == endIndex:
                lastTime = end - t
            else:
                lastTime = (i + 1) * 3600 - t
            
            # Use window-based indexing for estimates
            if i > maxIndex:
                hour_index = (minIndex + (i - minIndex) % (maxIndex - minIndex + 1)) % len(self.carbonIntensityList)
            else:
                hour_index = i % len(self.carbonIntensityList)
            
            carbonIntensity = self.carbonIntensityList[hour_index]
            energyKWh = (power / 1000.0) * (lastTime / 3600.0)
            emissions = energyKWh * carbonIntensity
            totalEmissions += emissions
            
            t = (i + 1) * 3600
        
        return totalEmissions

    def getCarbonAwareReward(self, powerSlot, jobList):
        """
        Calculate carbon-aware reward that considers job carbon consideration indices
        Higher carbon consideration = higher penalty for emissions
        Lower carbon consideration = lower penalty for emissions
        
        Args:
            powerSlot: power consumption slots
            jobList: list of jobs with their carbon_consideration indices
        
        Returns:
            Carbon-aware reward (negative value, higher magnitude = worse)
        """
        totalWeightedEmissions = 0
        totalEnergy = 0
        
        for i in range(len(powerSlot) - 1):
            power = powerSlot[i]['power']
            start = powerSlot[i]['timeSlot']
            end = powerSlot[i + 1]['timeSlot']
            lastTime = end - start
            
            energyKWh = (power / 1000.0) * (lastTime / 3600.0)
            totalEnergy += energyKWh
            
            emissions = self.getCarbonEmissions(power, start, end)
            
            # Calculate weighted emissions based on jobs running during this period
            carbonWeight = self.calculateCarbonWeight(jobList, start, end)
            weightedEmissions = emissions * carbonWeight
            totalWeightedEmissions += weightedEmissions
        
        if totalEnergy == 0:
            return 0
        else:
            # Return negative reward (penalty) - higher emissions = more negative
            avgWeightedCarbonIntensity = totalWeightedEmissions / totalEnergy
            # Normalize by max carbon intensity and convert to penalty
            return -(avgWeightedCarbonIntensity / MAX_CARBON_INTENSITY)
    
    def calculateCarbonWeight(self, jobList, start, end):
        """
        Calculate the carbon consideration weight for a time period
        based on jobs running during that time
        
        Args:
            jobList: list of jobs with carbon_consideration and timing info
            start, end: time period
            
        Returns:
            Weight factor (0.0 to 1.0, where 1.0 = highest carbon concern)
        """
        if not jobList:
            return 0.0  # No jobs, no carbon consideration
        
        runningJobs = []
        for job in jobList:
            jobStart = getattr(job, 'scheduled_time', -1)
            jobEnd = jobStart + getattr(job, 'request_time', 0) if jobStart != -1 else -1
            
            # Check if job overlaps with the time period
            if jobStart != -1 and jobStart < end and jobEnd > start:
                runningJobs.append(job)
        
        if not runningJobs:
            return 0.0  # No running jobs, no carbon consideration
        
        # Calculate average carbon consideration of running jobs
        totalCarbonConsideration = sum(getattr(job, 'carbon_consideration', 0) for job in runningJobs)
        avgCarbonConsideration = totalCarbonConsideration / len(runningJobs)
        
        # If all jobs have carbon consideration 0, return weight 0 (no carbon optimization)
        if avgCarbonConsideration == 0:
            return 0.0
        
        # Convert 0-4 scale to 0.2-1.0 weight scale for non-zero considerations
        # 1 (low concern) -> 0.2 weight (light penalty)
        # 4 (highest concern) -> 1.0 weight (full penalty)
        weight = 0.2 + ((avgCarbonConsideration - 1) / 3.0) * 0.8
        return max(0.2, weight)  # Ensure minimum 0.2 for any non-zero consideration

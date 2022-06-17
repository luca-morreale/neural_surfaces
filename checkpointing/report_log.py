
import math


class ReportData():
    ### Save data for report
    def __init__(self):
        self.report_data = {}

    def add_log(self, key, title):
        self.report_data[key] = {}
        self.report_data[key]['title'] = title
        self.report_data[key]['data']  = {}

    def add_entry(self, key, data, uid):
        self.report_data[key]['data'][uid] = data.detach()


class DataMeter():
    def __init__(self):
        self.report_data = {}

    def add_entry(self, key, data):
        if key not in self.report_data:
            self.report_data[key] = []
        if type(data) == float:
            self.report_data[key].append(math.log10(data))
        else:
            self.report_data[key].append(data.detach().cpu().log10().item())

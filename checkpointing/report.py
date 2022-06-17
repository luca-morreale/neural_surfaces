
from utils import create_model_summary

from .mixin import Mixin


class ReportCheckpointing(Mixin):

    def check_imports(self):
        ### check if library exists, if does not remove function
        try:
            from netvision import HtmlGenerator
        except ImportError as err:
            print('Error missing library ' + str(err))
            self.save_report = self.empty_function


    def save_report(self, tables_data, model, training_time, checkpoint_dir, prefix=''):
        ### Generate html with reports of the experiments

        from netvision import HtmlGenerator

        out_folder = self.compose_out_folder(checkpoint_dir, ['html'])
        filename = '{}/{}report.html'.format(out_folder, prefix)

        webpage = HtmlGenerator(path=filename, title="Report", local_copy=True)


        for _, table in tables_data.items():

            ###### skip empty blocks
            if len(table['data'].items()) == 0:
                continue

            ###### reconstruction losses
            webpage.add_title(table['title'])
            table_html = webpage.add_table()

            table_html.add_column("")
            table_html.add_column("Num points")
            table_html.add_column("Min")
            table_html.add_column("Mean")
            table_html.add_column("Median")
            table_html.add_column("Max")
            table_html.add_column("Sum")
            # table_html.add_column("Chart")

            # for each reconstruction element (eg patch) log name (key) and value
            for k, rec_el in table['data'].items():
                # curve_recons = webpage.chart({'distance':rec_el.tolist()}, title="Reconstruction quality")
                table_html.add_row([k, rec_el.size(0), rec_el.min().item(), rec_el.mean().item(), \
                                    rec_el.median().item(), rec_el.max().item(), rec_el.sum().item()])

            webpage.return_html()

        ###### Save model & training info into a different html file
        filename = '{}/info.html'.format(out_folder)
        webpage = HtmlGenerator(path=filename, title="Report", local_copy=True)

        def create_parameter_table(info, level):
            webpage.add_title(f"Model Info {level}")
            table_html = webpage.add_table()
            table_html.add_column("")
            table_html.add_column("Type")
            table_html.add_column("# params")
            table_html.add_column("Size (MB)")
            table_html.add_column("% of model")

            tot_params = info[''].num_parameters

            for name, layer_info in info.items():
                num_params = layer_info.num_parameters
                table_html.add_row([name, layer_info.layer_type, num_params, self.params_to_mb(num_params), num_params / tot_params * 100.0])

        summary = create_model_summary(model)
        for level in range(len(summary)):
            create_parameter_table({k:v for d in summary[:level+1] for k, v in d.items()}, level)


        webpage.return_html()

        ###### training time
        webpage.add_title("Training time")
        duration_dict = self.getDuration(training_time)

        table_html = webpage.add_table()
        table_html.add_column("years")
        table_html.add_column("days")
        table_html.add_column("hours")
        table_html.add_column("minutes")
        table_html.add_column("seconds")
        table_html.add_row([duration_dict['years'], duration_dict['days'], duration_dict['hours'], duration_dict['minutes'], duration_dict['seconds']])
        webpage.return_html()

        ## Do I want to save meshes as well?
        ## they are all in the path mesh anyway (only works with obj written)


    def params_to_mb(self, num_params):
        return (num_params * 4 / 1024) / 1024

    def count_params(self, params_list):
        count = 0
        for param in params_list:
            count += param.numel()
        return count

    def getDuration(self, delta, interval = "default"):

        # Returns a duration as specified by variable interval
        # Functions, except totalDuration, returns [quotient, remainder]

        duration_in_s = delta.total_seconds()

        def years():
            return divmod(duration_in_s, 31536000) # Seconds in a year=31536000.

        def days(seconds = None):
            return divmod(seconds if seconds != None else duration_in_s, 86400) # Seconds in a day = 86400

        def hours(seconds = None):
            return divmod(seconds if seconds != None else duration_in_s, 3600) # Seconds in an hour = 3600

        def minutes(seconds = None):
            return divmod(seconds if seconds != None else duration_in_s, 60) # Seconds in a minute = 60

        def seconds(seconds = None):
            if seconds != None:
                return divmod(seconds, 1)
            return duration_in_s

        y = years()
        d = days(y[1]) # Use remainder to calculate next variable
        h = hours(d[1])
        m = minutes(h[1])
        s = seconds(m[1])

        return {
            'years': int(y[0]),
            'days': int(d[0]),
            'hours': int(h[0]),
            'minutes': int(m[0]),
            'seconds': int(s[0]),
            'default': "Time between dates: {} years, {} days, {} hours, {} minutes and {} seconds".format(int(y[0]), int(d[0]), int(h[0]), int(m[0]), int(s[0]))
        }

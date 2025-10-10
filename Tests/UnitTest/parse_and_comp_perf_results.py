#!/usr/bin/env python3

import json
import argparse
import pathlib
import sys

def comp_indv_json(cur, golden, cyc_delt, mac_delt):
    pmu = "PMU"
    if (pmu in golden.keys()):
        for key in golden["PMU"].keys():
            if ((abs(cur["PMU"][key] - golden["PMU"][key])) > (mac_delt / 100) * golden["PMU"][key]):
                return False

    if (((cur["DWT_CYCCNT"] - golden["DWT_CYCCNT"])) > (cyc_delt / 100) * golden["DWT_CYCCNT"]):
        return False
    else:
        return True
    
def comp_strict_indv_json(cur, golden):
    return comp_indv_json(cur, golden, 0, 1)

def comp_and_store(args):
    perf_test_results_json = args.perf_test_results_json.read_text()
    perf_json_to_parse = json.loads(perf_test_results_json)

    perf_golden_results_json = args.perf_golden_results_json.read_text()
    perf_golden_to_parse = json.loads(perf_golden_results_json)

    retrograde = 0
    perf_results_by_tc = {}
    perf_golden_by_tc = {}
    perf_regressed = []
    perf_golden_list = []

    for indv_json in perf_json_to_parse:
        perf_results_by_tc[indv_json["op"]] = indv_json

    for golden_json in perf_golden_to_parse:
        perf_golden_by_tc[golden_json["op"]] = golden_json

    for indv_json in perf_json_to_parse:
        regressed_json = {}
        if (indv_json["op"] in perf_golden_by_tc):
            if (comp_indv_json(indv_json, perf_golden_by_tc[indv_json["op"]], args.cyc_delt, args.mac_delt) == False):
                regressed_json["failed_tc"] = indv_json["op"]
                regressed_json["current_perf"] = indv_json
                regressed_json["golden_perf"] = perf_golden_by_tc[indv_json["op"]]
                perf_regressed.append(regressed_json)
                retrograde += 1
            elif (comp_strict_indv_json(indv_json, perf_golden_by_tc[indv_json["op"]]) == True):
                perf_golden_by_tc[indv_json["op"]] = perf_results_by_tc[indv_json["op"]]
        else:
            perf_golden_by_tc[indv_json["op"]] = perf_results_by_tc[indv_json["op"]]
    
    err_filename = str(args.perf_test_results_json)
    err_filename = err_filename[:(len(err_filename) - 5)] + "_err.json"

    ref_filename = str(args.perf_golden_results_json)

    with open(err_filename, "w") as file:
        json.dump(perf_regressed, file)

    
    for key in perf_golden_by_tc:
        perf_golden_list.append(perf_golden_by_tc[key])

    with open(ref_filename, "w") as file:
        json.dump(perf_golden_list, file)

    return retrograde
    
def print_err_json(args):
    err_filename = str(args.perf_test_results_json)
    err_filename = err_filename[:(len(err_filename) - 5)] + "_err.json"
    with open(err_filename, "r") as file:
        err_json = json.load(file)
        print(json.dumps(err_json, indent=4))

def main():
    parser = argparse.ArgumentParser(
        prog="parse_and_comp_perf_results",
        description="Parses performance test json, compares it against, and updates existing data if applicable."
    )

    parser.add_argument("-p",
                        "--perf_test_results_json",
                        type=pathlib.Path,
                        default="./perf_results_cortex-m55.json",
                        help="Path to a json performance results file")
    
    parser.add_argument("-g",
                        "--perf_golden_results_json",
                        type=pathlib.Path,
                        default="./perf_results_cortex-m55_ref.json",
                        help="Path to a json previous best performance results file")

    parser.add_argument("-c",
                        "--cyc_delt",
                        type=float,
                        default=5.0,
                        help="Largest acceptable percent decrease (float) in cycles for performance")
    
    parser.add_argument("-m",
                        "--mac_delt",
                        type=float,
                        default=1.0,
                        help="Largest acceptable percent delta (float) within expected MACs for performance")

    args = parser.parse_args()

    perf_pass = comp_and_store(args)

    if (perf_pass == 0):
        print("No Failures")
    else:
        print(f"{perf_pass} Failures")
        print_err_json(args)

    return perf_pass

if __name__ == '__main__':
    if (main() == 0):
        sys.exit(0)
    else:
        sys.exit(1)



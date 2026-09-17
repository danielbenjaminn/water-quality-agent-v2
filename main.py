import argparse
from water_quality_agent.llm import build_llm
from water_quality_agent.agent.runtime import WaterQualityAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    agent = WaterQualityAgent(build_llm())
    ingestion = agent.load_csv(args.csv)
    print(ingestion["report_text"])

    if not ingestion["valid"]:
        return

    if args.question:
        result = agent.ask(args.question)
        print("\n" + result["messages"][-1].content)


if __name__ == "__main__":
    main()

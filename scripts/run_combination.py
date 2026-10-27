#!/usr/bin/env python3
from temporal_dte.combine import evaluate

if __name__ == "__main__":
    _, _, family, overall = evaluate()
    print("Family summary\n", family.to_string(index=False))
    print("\nOverall\n", overall.to_string(index=False))

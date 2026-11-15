from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trainer.core.settings import get_settings
from trainer.data.issda import download_cer_dataset
from trainer.services.pipeline import PipelineService


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Trainer worker for the federated smart energy prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("worker", help="Run the background worker loop")
    bootstrap = subparsers.add_parser("bootstrap-fixture", help="Generate fixture data, train, and publish a demo model")
    bootstrap.add_argument("--households", type=int, default=settings.synthetic_dataset_households)
    bootstrap.add_argument("--days", type=int, default=settings.synthetic_dataset_days)
    bootstrap.add_argument("--seed", type=int, default=settings.synthetic_dataset_seed)
    bootstrap.add_argument("--no-secure-aggregation", action="store_true")
    bootstrap.add_argument("--secure-aggregation-degree", type=int, default=2)
    bootstrap.add_argument("--secure-aggregation-mask-scale", type=float, default=4.0)
    evaluate_fixture = subparsers.add_parser("evaluate-fixture", help="Run fixture DP-on and DP-off experiments and write a comparison report")
    evaluate_fixture.add_argument("--households", type=int, default=settings.synthetic_dataset_households)
    evaluate_fixture.add_argument("--days", type=int, default=settings.synthetic_dataset_days)
    evaluate_fixture.add_argument("--seed", type=int, default=settings.synthetic_dataset_seed)
    evaluate_fixture.add_argument("--weather-household-limit", type=int, default=None)
    evaluate_fixture.add_argument("--rounds", type=int, default=5)
    evaluate_fixture.add_argument("--local-epochs", type=int, default=2)
    evaluate_fixture.add_argument("--batch-size", type=int, default=64)
    evaluate_fixture.add_argument("--learning-rate", type=float, default=0.001)
    evaluate_fixture.add_argument("--noise-multiplier", type=float, default=1.1)
    evaluate_fixture.add_argument("--clipping-norm", type=float, default=1.0)
    evaluate_fixture.add_argument("--no-secure-aggregation", action="store_true")
    evaluate_fixture.add_argument("--secure-aggregation-degree", type=int, default=2)
    evaluate_fixture.add_argument("--secure-aggregation-mask-scale", type=float, default=4.0)
    evaluate_fixture.add_argument("--publish-dp-on", action="store_true")
    download_cer = subparsers.add_parser("download-cer", help="Download CER docs and, with an ISSDA API key, restricted data files")
    download_cer.add_argument("--target-dir", type=Path, default=None)
    download_cer.add_argument("--api-key", default=None)
    bootstrap_cer = subparsers.add_parser("bootstrap-cer", help="Import CER data, sync weather, train, approve, and publish")
    bootstrap_cer.add_argument("--profile", choices=["cer-dev", "cer-full"], default="cer-dev")
    bootstrap_cer.add_argument("--dataset-path", type=Path, default=None)
    bootstrap_cer.add_argument("--sample-households", type=int, default=12)
    bootstrap_cer.add_argument("--weather-household-limit", type=int, default=None)
    bootstrap_cer.add_argument("--rounds", type=int, default=5)
    bootstrap_cer.add_argument("--local-epochs", type=int, default=2)
    bootstrap_cer.add_argument("--batch-size", type=int, default=64)
    bootstrap_cer.add_argument("--learning-rate", type=float, default=0.001)
    bootstrap_cer.add_argument("--noise-multiplier", type=float, default=1.1)
    bootstrap_cer.add_argument("--clipping-norm", type=float, default=1.0)
    bootstrap_cer.add_argument("--no-secure-aggregation", action="store_true")
    bootstrap_cer.add_argument("--secure-aggregation-degree", type=int, default=2)
    bootstrap_cer.add_argument("--secure-aggregation-mask-scale", type=float, default=4.0)
    bootstrap_cer.add_argument("--no-dp", action="store_true")
    bootstrap_smart = subparsers.add_parser(
        "bootstrap-smart-meters",
        help="Import smart meter data, sync weather, train, approve, and publish",
    )
    bootstrap_smart.add_argument("--dataset-path", type=Path, default=None)
    bootstrap_smart.add_argument("--sample-households", type=int, default=12)
    bootstrap_smart.add_argument("--weather-household-limit", type=int, default=None)
    bootstrap_smart.add_argument("--rounds", type=int, default=5)
    bootstrap_smart.add_argument("--local-epochs", type=int, default=2)
    bootstrap_smart.add_argument("--batch-size", type=int, default=64)
    bootstrap_smart.add_argument("--learning-rate", type=float, default=0.001)
    bootstrap_smart.add_argument("--noise-multiplier", type=float, default=1.1)
    bootstrap_smart.add_argument("--clipping-norm", type=float, default=1.0)
    bootstrap_smart.add_argument("--no-secure-aggregation", action="store_true")
    bootstrap_smart.add_argument("--secure-aggregation-degree", type=int, default=2)
    bootstrap_smart.add_argument("--secure-aggregation-mask-scale", type=float, default=4.0)
    bootstrap_smart.add_argument("--no-dp", action="store_true")
    args = parser.parse_args()

    service = PipelineService()
    if args.command == "worker":
        service.worker_loop()
    elif args.command == "bootstrap-fixture":
        result = service.bootstrap_profile(
            source_type="fixture",
            profile="fixture",
            sample_households=args.households,
            synthetic_days=args.days,
            synthetic_seed=args.seed,
            run_config_overrides={
                "secure_aggregation_enabled": not args.no_secure_aggregation,
                "secure_aggregation_degree": args.secure_aggregation_degree,
                "secure_aggregation_mask_scale": args.secure_aggregation_mask_scale,
            },
            reset_existing=True,
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "evaluate-fixture":
        result = service.evaluate_fixture_profile(
            households=args.households,
            synthetic_days=args.days,
            synthetic_seed=args.seed,
            weather_household_limit=args.weather_household_limit,
            run_config_overrides={
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "secure_aggregation_enabled": not args.no_secure_aggregation,
                "secure_aggregation_degree": args.secure_aggregation_degree,
                "secure_aggregation_mask_scale": args.secure_aggregation_mask_scale,
                "noise_multiplier": args.noise_multiplier,
                "clipping_norm": args.clipping_norm,
            },
            publish_dp_on=args.publish_dp_on,
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "download-cer":
        target_dir = args.target_dir or (settings.raw_data_dir / "issda")
        api_key = args.api_key or os.getenv("ISSDA_DATAVERSE_API_KEY")
        result = download_cer_dataset(target_dir, api_key=api_key)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "bootstrap-cer":
        result = service.bootstrap_profile(
            source_type="cer",
            profile=args.profile,
            dataset_path=args.dataset_path,
            sample_households=args.sample_households,
            weather_household_limit=args.weather_household_limit,
            run_config_overrides={
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "secure_aggregation_enabled": not args.no_secure_aggregation,
                "secure_aggregation_degree": args.secure_aggregation_degree,
                "secure_aggregation_mask_scale": args.secure_aggregation_mask_scale,
                "dp_enabled": not args.no_dp,
                "noise_multiplier": args.noise_multiplier,
                "clipping_norm": args.clipping_norm,
            },
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "bootstrap-smart-meters":
        result = service.bootstrap_profile(
            source_type="smart-meters",
            profile="smart-meters",
            dataset_path=args.dataset_path,
            sample_households=args.sample_households,
            weather_household_limit=args.weather_household_limit,
            run_config_overrides={
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "secure_aggregation_enabled": not args.no_secure_aggregation,
                "secure_aggregation_degree": args.secure_aggregation_degree,
                "secure_aggregation_mask_scale": args.secure_aggregation_mask_scale,
                "dp_enabled": not args.no_dp,
                "noise_multiplier": args.noise_multiplier,
                "clipping_norm": args.clipping_norm,
            },
        )
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

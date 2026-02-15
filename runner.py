# runner.py
# CL, Local-only, FL (FedAvg/FedProx/FedBN/SCAFFOLD)
# Seeds + bootstrap CIs (no K-fold)
# Validation is created from train.csv via StratifiedShuffleSplit when --val_from_train is given.
# CSV schema expected: dataset, img, label, roi_png_path



# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cl','local','fl'], required=True)
    ap.add_argument('--algo', choices=['fedavg','fedprox','fedbn','scaffold'], default='scaffold')
    ap.add_argument('--model', choices=['resnet50','swin_t','convnext_b'], default='resnet50')
    ap.add_argument('--img_size', type=int, default=224)

    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, default=None)
    # may be ignored when --val_from_train
    ap.add_argument('--val_from_train', action='store_true', help="Create validation split from train.csv")
    ap.add_argument('--val_ratio', type=float, default=0.15, help="Validation ratio from train.csv when --val_from_train")
    ap.add_argument('--test_csv', type=str, default=None)    # optional combined test
    ap.add_argument('--tests', action='append', help="Optional multiple test sets as name:path, e.g., --tests cbis:combined\\cbis_test.csv")

    ap.add_argument('--dataset_col', type=str, default='dataset')
    ap.add_argument('--path_col', type=str, default='roi_png_path')
    ap.add_argument('--label_col', type=str, default='label')

    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=100)       # CL/local epochs
    ap.add_argument('--rounds', type=int, default=100)       # FL rounds
    ap.add_argument('--local_epochs', type=int, default=1)   # FL local steps
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--mu', type=float, default=0.01)        # FedProx μ
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=str, default='results')

    ap.add_argument('--early_stop', action='store_true', help="Enable early stopping based on validation score")
    ap.add_argument('--patience', type=int, default=10, help="Epochs/rounds without improvement before stopping")
    ap.add_argument('--min_delta', type=float, default=0.0, help="Required minimal improvement to reset patience")

    # NEW: export per-client personalized checkpoints (global + client BN)
    ap.add_argument('--export_personalized', action='store_true',
                    help="(FedBN) Export full per-client personalized checkpoints (global + client BN)")
    
    ap.add_argument('--n_clients_if_single', type=int, default=2,
                        help="If train split contains a single dataset, stratify by label into this many pseudo-clients (e.g., 2).")
    ap.add_argument('--balance_vindr_negatives', action='store_true',
                        help="On mixed ViNDR+CBIS train data, drop ViNDR label=0 rows to match total size to CBIS (per seed).")

    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for seed in range(args.seeds):
        # Create per-seed outdir and (if requested) per-seed train/val split from train.csv
        outdir = Path(args.out)/f"{args.mode}_{args.algo}_{args.model}_sz{args.img_size}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)

        if args.val_from_train:
            
            tr_csv, va_csv = make_val_from_train_by_dataset_and_patient(
                train_csv=args.train_csv,
                val_ratio=args.val_ratio,
                seed=seed,
                label_col=args.label_col,
                dataset_col=args.dataset_col,
                patient_col = 'img',
                outdir=outdir
            )

        else:
            assert args.val_csv is not None, "Provide --val_csv or use --val_from_train"
            tr_csv, va_csv = args.train_csv, args.val_csv

        # ---------- Optional re-shaping of TRAIN split ONLY ----------
        # (A) Balance ViNDR vs CBIS by downsampling ViNDR negatives to match total counts
        if args.balance_vindr_negatives:
            _df_tr = pd.read_csv(tr_csv)
            _df_tr_bal = balance_vindr_by_dropping_negatives_to_match(
                _df_tr, dataset_col=args.dataset_col, label_col=args.label_col, seed=seed
            )
            if len(_df_tr_bal) != len(_df_tr):
                tr_bal_csv = Path(outdir) / f"tmp_train_balanced_seed{seed}.csv"
                _df_tr_bal.to_csv(tr_bal_csv, index=False)
                tr_csv = str(tr_bal_csv)

        # (B) If FL over a single dataset (e.g., only CBIS), split into K pseudo-clients stratified by label
        if args.mode == 'fl' and args.n_clients_if_single and args.n_clients_if_single >= 2:
            _df_tr = pd.read_csv(tr_csv)
            if _df_tr[args.dataset_col].dropna().astype(str).nunique() == 1:
                _df_tr_clients = stratify_single_dataset_into_n_clients(
                    _df_tr, dataset_col=args.dataset_col, label_col=args.label_col,
                    n_clients=args.n_clients_if_single, seed=seed
                )
                # Save modified train CSV with pseudo-client dataset names
                tr_clients_csv = Path(outdir) / f"tmp_train_{args.n_clients_if_single}clients_seed{seed}.csv"
                _df_tr_clients.to_csv(tr_clients_csv, index=False)
                tr_csv = str(tr_clients_csv)
        if args.mode=='cl':
            run_centralized(args, seed, device, outdir, tr_csv, va_csv)
        elif args.mode=='local':
            run_local_only(args, seed, device, outdir, tr_csv, va_csv)
        else:
            run_federated(args, seed, device, outdir, tr_csv, va_csv)

if __name__ == "__main__":
    main()


"""
python runner.py --mode fl --algo fedbn --model resnet50 \
  --train_csv combined/cbis_train.csv \
  --val_from_train \
  --rounds 100 --local_epochs 1 --seeds 3 \
  --n_clients_if_single 2 \
  --export_personalized \
  --out results_fl_fedbnv6

 python runner.py --mode fl --algo fedavg --model resnet50 --train_csv combined/cbis_train.csv --val_from_train --rounds 100 --local_epochs 1 --seeds 3 --n_clients_if_single 2 --out results_fl_fedavg_resnet_homm_cbis


python runner.py --mode fl --algo fedbn --model resnet50 \
  --train_csv combined/vindr_cbis_train.csv \
  --val_from_train \
  --test_csv combined/vindr_cbis_test.csv \
  --rounds 100 --local_epochs 1 --seeds 3 \
  --balance_vindr_negatives \
  --export_personalized \
  --out results_fl_fedbnv6

python runner.py --mode fl --algo fedavg --model resnet50 --train_csv combined/vindr_cbis_train.csv --val_from_train --test_csv combined/vindr_cbis_test.csv --rounds 100 --local_epochs 1 --seeds 3 --balance_vindr_negatives  --out results_fl_fedavg_resnet_blc
python runner.py --mode cl --algo fedavg --model resnet50 --train_csv combined/vindr_cbis_train.csv --val_from_train --test_csv combined/vindr_cbis_test.csv --rounds 100 --local_epochs 1 --seeds 3 --balance_vindr_negatives  --out results_cl_resnet_blc




"""

\begin{table*}[htbp]
    \begin{center}
    \caption{Segmentation performance across five thyroid ultrasound datasets. Upper block, Dice score; lower block, 95th percentile Hausdorff distance (HD95). Values are reported with 95\% confidence intervals.}
    \label{tab:seg_performance}
    \begin{tabular}{cccccc}
        \toprule
        \textbf{Method} & \textbf{TN3K} & \textbf{DDTI} & \textbf{ThyroidXL} & \textbf{PKTN} & \textbf{TN5K}\\
        \midrule
        % 切换transunet的引用为mia的版本
        TransUnet~\cite{chen2024transunet} & $81.84 \pm 1.62$ & $76.96 \pm 4.79$ & $85.75 \pm 0.57$ & $76.89 \pm 3.56$ & $78.54 \pm 1.51$ \\
        MedSegX~\cite{zhang2025generalist} & $83.93 \pm 0.79$ & $85.34 \pm 0.42$ & $79.98 \pm 0.36$ & $80.63 \pm 0.42$ & $83.10 \pm 0.48$\\
        MedSAM2~\cite{ma2025medsam2} & $84.47 \pm 1.02 $ & $84.90 \pm 1.51 $ & $83.74 \pm 0.46 $ & $80.71 \pm 0.98 $ & $81.22 \pm 1.14$ \\
        UltraFedFM~\cite{jiang2025pretraining} & $81.18 \pm 1.46$ & $75.55 \pm 1.57$ & $84.70 \pm 0.53$ & $75.31 \pm 1.12$ & $77.13 \pm 1.38$ \\
        \rowcolor{lightgray}
        \textbf{ThyroidXAgent} & $85.28 \pm 1.28$ & $85.16 \pm 2.71$ & $87.58 \pm 0.44$ & $82.96 \pm 1.98$ & $83.26 \pm 1.34$ \\
        \midrule
        % 切换transunet的引用为mia的版本
        TransUnet~\cite{chen2024transunet} & $39.27 \pm 5.52$ & $25.46 \pm 5.54$ & $27.42 \pm 1.34$ & $36.88 \pm 9.66$ & $32.32 \pm 3.43$ \\
        MedSegX~\cite{zhang2025generalist} & $10.95 \pm 0.64$ & $10.44 \pm 0.48$ & $11.07 \pm 0.32$ & $10.83 \pm 0.70$ & $11.76 \pm 0.76$\\
        MedSAM2~\cite{ma2025medsam2} & $11.51 \pm 1.53 $ & $9.68 \pm 2.08 $ & $6.91 \pm 0.44 $ & $10.69 \pm 2.34 $ & $10.94 \pm 1.12 $ \\
        UltraFedFM~\cite{jiang2025pretraining} & $14.98 \pm 2.10 $ & $18.10 \pm 1.52$ & $8.10 \pm 0.58$ & $16.08 \pm 1.67$ & $14.96 \pm 1.65$ \\
        \rowcolor{lightgray}
        \textbf{ThyroidXAgent} & $10.31 \pm 1.70$ & $9.44 \pm 3.54$ & $5.43 \pm 0.53$ & $9.01 \pm 3.58$ & $10.12 \pm 1.23$ \\
        \bottomrule
    \end{tabular}
  \end{center}
\end{table*}

\begin{table*}[!htp]
    \begin{center}
    \caption{Classification performance across four thyroid ultrasound datasets. Upper block, area under the receiver operating characteristic curve (AUROC); lower block, area under the precision--recall curve (AUPRC). Values are reported with 95\% confidence intervals.}
    \label{tab:cls_performance}
    \begin{tabular}{ccccc}
        \toprule
        \textbf{Method} & \textbf{TN3K} & \textbf{DDTI} & \textbf{ThyroidXL} & \textbf{TN5K}\\
        \midrule
        ResNet-50~\cite{he2016deep} & $0.7674 \pm 0.0394$ & $0.6704 \pm 0.0842$ & $0.9044 \pm 0.0118$ & $0.9322 \pm 0.0168$ \\
        RepViT~\cite{wang2023repvit} & $0.5556 \pm 0.0463$ & $0.6162 \pm 0.0804$ & $0.7774 \pm 0.0188$ & $0.6603 \pm 0.0375$\\
        LSNet~\cite{wang2025lsnet} & $0.8095 \pm 0.0333$ & $0.7581 \pm 0.0658$ & $0.9178 \pm 0.0114$ & $0.9091 \pm 0.0201$\\
        UltraFedFM~\cite{jiang2025pretraining} & $0.8461 \pm 0.0697$ & $0.7518 \pm 0.1712$ & $0.9239 \pm 0.0104$ & $0.9298 \pm 0.0175$\\
        MedGemma~\cite{sellergren2025medgemma} & $0.8492 \pm 0.0305$ & $0.8255 \pm 0.0650$ & $0.9371 \pm 0.0095$ & $0.9442 \pm 0.0156$\\
        Qwen3-VL-8B-Instruct~\cite{bai2025qwen3} & $0.8237 \pm 0.0328$ & $0.7361 \pm 0.0692$ & $0.9050 \pm 0.0115$ & $0.9214 \pm 0.0187$\\
        % 引用gpt5的system card
        GPT-5.5~\cite{openai2025gpt5systemcard} & $0.6924 \pm 0.0421$ & $0.6346 \pm 0.0914$ & $0.7059 \pm 0.0469$ & $0.7737 \pm 0.0996$\\
        Gemini-3.1-Pro~\cite{comanici_gemini_2025} & $0.6587 \pm 0.0455$ & $0.6156 \pm 0.1308$ & $0.6246 \pm 0.0640$ & $0.6873 \pm 0.0691$\\
        \rowcolor{lightgray}
        \textbf{ThyroidXAgent} & $0.8692 \pm 0.0349$ & $0.7991 \pm 0.0741$ & $0.9676 \pm 0.0066$ & $0.9472 \pm 0.0152$\\
        \midrule
        ResNet-50~\cite{he2016deep} & $0.6882 \pm 0.0632$ & $0.3755 \pm 0.1176$ & $0.8882 \pm 0.0174$ & $0.9674 \pm 0.0268$ \\
        RepViT~\cite{wang2023repvit} & $0.4275 \pm 0.0528$ & $0.3924 \pm 0.0933$ & $0.7161 \pm 0.0276$ & $0.8403 \pm 0.0216$\\
        LSNet~\cite{wang2025lsnet} & $0.7581 \pm 0.0452$ & $0.4180 \pm 0.1410$ & $0.9040 \pm 0.0142$ & $0.9551 \pm 0.0134$\\
        UltraFedFM~\cite{jiang2025pretraining} & $0.8531 \pm 0.0284$ & $0.4487 \pm 0.1452$ & $0.9354 \pm 0.0114$ & $0.8422 \pm 0.0421$\\
        MedGemma~\cite{sellergren2025medgemma} & $0.8047 \pm 0.0430$ & $0.5537 \pm 0.1663$ & $0.9201 \pm 0.0139$ & $0.9747 \pm 0.0084$\\
        Qwen3-VL-8B-Instruct~\cite{bai2025qwen3} & $0.7617 \pm 0.0511$ & $0.4112 \pm 0.1415$ & $0.8787 \pm 0.0379$ & $0.9636 \pm 0.0106$\\
        % 引用gpt5的system card
        GPT-5.5~\cite{openai2025gpt5systemcard} & $0.6627 \pm 0.0633$ & $0.3578 \pm 0.1089$ & $0.6237 \pm 0.0666$ & $0.8920 \pm 0.0316$\\
        Gemini-3.1-Pro~\cite{comanici_gemini_2025} & $0.6205 \pm 0.0587$ & $0.3924 \pm 0.1527$ & $0.4914 \pm 0.0841$ & $0.8462 \pm 0.0446$\\
        \rowcolor{lightgray}
        \textbf{ThyroidXAgent} & $0.8545 \pm 0.0600$ & $0.5863 \pm 0.1380$ & $0.9653 \pm 0.0078$ & $0.9752 \pm 0.0089$\\
        \bottomrule
    \end{tabular}
  \end{center}
\end{table*}


\begin{table*}[!htp]
    \begin{center}
    \caption{Performance of methods on lymph node metastasis (LNM) prediction and follicular thyroid carcinoma versus papillary thyroid carcinoma (FTC/PTC) subtype classification. Values are reported as AUROC and AUPRC with 95\% confidence intervals. Em dashes indicate tasks not evaluated for a given method.}
    \label{tab:Malignant_images_tasks_performance}
    \begin{tabular}{ccccc}
        \toprule
        \multirow{2}{*}{\textbf{Method}} &
        \multicolumn{2}{c}{\textbf{Lymph Node Metastasis}} &
        \multicolumn{2}{c}{\textbf{FTC/PTC subtype}} \\
        & AUROC & AUPRC & AUROC & AUPRC \\
        \midrule
        RepViT~\cite{wang2023repvit} & $0.7905 \pm 0.0676$ & $0.8152 \pm 0.0638$ & $0.6419 \pm 0.0839$ & $0.6297 \pm 0.0942$\\
        LSNet~\cite{wang2025lsnet} & $0.5878 \pm 0.0865$ & $0.6301 \pm 0.0875$ & $0.4858 \pm 0.0925$ & $0.4845 \pm 0.0908$\\
        UltraFedFM~\cite{jiang2025pretraining} & $0.7757 \pm 0.0731$ & $0.7902 \pm 0.0845$ & $0.7365 \pm 0.0744$ & $0.7582 \pm 0.0824$\\
        MedGemma~\cite{sellergren2025medgemma} & $0.8403 \pm 0.0461$ & $0.8585 \pm 0.0566$ & $0.6598 \pm 0.0824$ & $0.6142 \pm 0.1056$\\
        Qwen3-VL-8B-Instruct~\cite{bai2025qwen3} & $0.8070 \pm 0.0632$ & $0.8055 \pm 0.0800 $ & $0.6056 \pm 0.0866$ & $0.5539 \pm 0.1118$\\
        % 引用gpt5的system card
        GPT-5.5~\cite{openai2025gpt5systemcard} & $0.8410 \pm 0.0575$ & $0.8629 \pm 0.0533 $ & $0.1604 \pm 0.0706$ & $0.3638 \pm 0.0847$\\
        Gemini-3.1-Pro~\cite{comanici_gemini_2025} & $0.5414 \pm 0.0736$ & $0.5492 \pm 0.0915 $ & $0.3324 \pm 0.0872$ & $0.4187 \pm 0.0837$\\
        LLNM-Net~\cite{shen2025explainable} & $0.7665 \pm 0.0692$ & $0.7363 \pm 0.0849 $ & --- & ---\\
        Tiger-Model~\cite{dai2025improving} & --- & --- & $0.7136 \pm 0.0814$ & $0.7117 \pm 0.1101 $\\
        \rowcolor{lightgray} % 设置背景色为浅灰色
        \textbf{ThyroidXAgent} & $0.8642 \pm 0.0550$ & $0.8808 \pm 0.0537 $ & $0.8053 \pm 0.0599$ & $0.7863 \pm 0.0793$\\
        \bottomrule
    \end{tabular}
  \end{center}
\end{table*}

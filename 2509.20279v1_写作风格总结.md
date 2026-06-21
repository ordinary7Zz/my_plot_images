# `2509.20279v1.pdf` 写作风格总结（四分版）

## 说明

- **目标**：把这篇论文拆成更适合模仿的四类风格：`正文叙述`、`主文图注`、`补充材料叙述`、`补充图注`。
- **适用场景**：你现在主要负责 **正文** 和 **图的 caption**，所以这份文档会优先告诉你主文该怎么写，同时把补充材料的风格单独拎出来，避免两者混写。
- **出处说明**：所有原文依据都来自提取后的 `2509.20279v1.md`。PDF 抽取造成了少量单词粘连，但**不影响句式、结构和叙述策略判断**。
- **补充说明**：当前 PDF 中能稳定识别出的补充材料，主要是 `Supplementary Figure` caption，以及正文里对补充图的引用方式；因此“补充材料叙述风格”的总结，主要建立在这些文本证据之上。

## 一句话结论

这篇论文最值得模仿的，不是某几个华丽词汇，而是它把不同部分写成了不同功能层：**正文负责讲故事和立贡献，主文图注负责高密度导航，补充材料负责补证据和补细节，补充图注负责把额外验证解释成“为什么主文可信”**。

---

## 一、整篇论文共有的底层风格

### 1. 先立“大问题”和“应用价值”，再立方法

- **风格总结**：作者不会一上来介绍自己的系统，而是先把研究问题放进更大的临床和科学语境里。整篇论文的底层气质是：**先证明问题重要，再证明自己值得被读**。
- **你写时可模仿**：正文和摘要都尽量先回答“为什么重要”，再回答“你做了什么”。
- **原文来源**：`2509.20279v1.md:21-27`
  > “Medical image analysis provides the computational foundation for understanding disease status and progression. It is essential for clinical decision-making, treatment planning, and, most importantly, advancing scientific discovery.”
  >
  > “At present, no single model reliably addresses the full spectrum of clinical imaging needs.”

### 2. 强调“临床可用 / 可解释 / 可转化”，不是只强调模型更强

- **风格总结**：文章反复把系统能力落到 clinical decision-making、trust、guideline-aligned、translational adoption 这些维度上，所以它不是一篇纯算法炫技文，而是一篇非常典型的生物医学综合型论文。
- **你写时可模仿**：如果你的研究靠近机制、病理、临床、生信、转化医学，就要让每个主要结果尽量落到“为什么对真实研究或应用重要”。
- **原文来源**：`2509.20279v1.md:17-17`, `2509.20279v1.md:31-35`, `2509.20279v1.md:1459-1465`
  > “Through experiments across diverse tasks, where clinically meaningful quantifications directly inform staging, prognosis, and treatment planning...”
  >
  > “guideline-aligned, and clinically meaningful applications...”
  >
  > “TissueLab bridges the gap between algorithmic innovation and clinical/translational adoption...”

### 3. 语气自信，但每个强结论都尽量有数字、机制或可视化托底

- **风格总结**：文中会频繁使用 `state-of-the-art`、`expert-level`、`trustworthy`、`clinically meaningful` 这类强词，但很少空喊口号，通常马上跟 benchmark、workflow、guideline、visualization 或 human feedback。
- **你写时可模仿**：可以写得有力量，但每个强判断后面都要跟证据。
- **原文来源**：`2509.20279v1.md:17-17`, `2509.20279v1.md:223-229`, `2509.20279v1.md:1465-1469`
  > “we show that TissueLab achieves state-of-the-art performance...”
  >
  > “reflecting expert-level accuracy...”
  >
  > “allowing clinicians to inspect and interpret the reasoning process...”

### 4. 高频使用并列结构，制造成熟、完整、期刊化的语感

- **风格总结**：这篇论文大量使用 `(i) ... (ii) ... (iii) ...`、`reliable, reproducible, and transparent` 这类并列结构。这样的句法非常 Nature 子刊：读起来像经过强压缩和强组织。
- **你写时可模仿**：当你总结贡献、方法原则、结果意义时，适度使用 2–4 项并列，会明显提升文章的“完成度”。
- **原文来源**：`2509.20279v1.md:29-31`, `2509.20279v1.md:1722-1722`
  > “TissueLab emphasizes four principles: (i) adaptivity...; (ii) co-evolution...; (iii) safety...; and (iv) community value...”
  >
  > “not only statistically valid but also clinically meaningful.”

---

## 二、正文叙述风格

### 1. `Abstract` 是高密度闭环：背景缺口 → 方法提出 → 机制概括 → 核心结果 → 影响意义

- **风格总结**：摘要不是松散概述，而是一段内快速完成完整闭环，让编辑和审稿人几十秒内就知道“问题、方案、证据、意义”。
- **你写时可模仿**：你的摘要里最好也有这五步，而且尽量不要分散在很多句里。
- **原文来源**：`2509.20279v1.md:17-17`
  > “Agentic AI is rapidly advancing in healthcare and biomedical research. However, in medical image analysis, their performance and adoption remains limited...”
  >
  > “Here we present ‘TissueLab’, a co-evolving agentic AI system...”
  >
  > “Through experiments across diverse tasks... we show that TissueLab achieves state-of-the-art performance...”
  >
  > “Released as a sustainable open-source ecosystem (tissuelab.org), we expect TissueLab to significantly advance and accelerate computational research and translational adoption...”

### 2. 引言非常强调“分层拆痛点”，不是一句话轻飘飘带过

- **风格总结**：引言把现有方法的问题拆成固定工具箱、幻觉、高维数据处理、专家无法干预、知识不沉淀等多个层次，使得研究缺口显得系统而非偶然。
- **你写时可模仿**：介绍不足时，尽量拆成 3–5 个互相独立但汇聚到你方案的缺点。
- **原文来源**：`2509.20279v1.md:27-31`
  > “Most agents depend on a fixed toolbox...”
  >
  > “hallucinations..., token overload, and attention dilution...”
  >
  > “clinicians and researchers are unable to intervene and refine intermediate tools dynamically...”

### 3. 方法提出句非常干净利落，典型句型就是 `To address these challenges, we introduce ...`

- **风格总结**：作者在完成问题铺垫后，会用一条极其标准的转折句把论文主角推出场，逻辑非常清晰。
- **你写时可模仿**：如果你的工作是系统、框架、模型、资源库，这种句子很适合直接借鉴。
- **原文来源**：`2509.20279v1.md:29-31`
  > “To address these challenges, we introduce TissueLab, a co-evolving agentic AI system designed for medical imaging analysis that continuously evolves with new tools and user feedback.”

### 4. `Results` 的基本节奏固定为：任务意义 → 数据/任务设置 → workflow → 结果数字 → 解释与对照

- **风格总结**：正文结果不是“我们做了实验，结果如下”，而是先说明任务为什么重要，再说明对象和数据，然后描述 agent/workflow，再给指标，再解释为什么这个结果站得住。
- **你写时可模仿**：每个结果子节尽量使用同一个骨架，这会让整篇主文非常稳。
- **原文来源**：`2509.20279v1.md:223-229`
  > “Many clinically important measurements in pathology remain labor-intensive...”
  >
  > “We first evaluated TissueLab on predicting tumor invasion depth...”
  >
  > “As illustrated in Figure 2b, TissueLab agent (TLAgent) designs a structured workflow...”
  >
  > “As a result, the predicted invasion depths were strongly correlated...”

### 5. 正文结果段喜欢“强结论 + 具体数字 + 基线对照 + 含义解释”绑定出现

- **风格总结**：作者很少只写“better than baseline”，而是把指标、比较对象、意义解释捆在一起，这会极大提高说服力。
- **你写时可模仿**：尽量写成“达到多少、超过谁、说明什么”，而不是只写“显著提升”。
- **原文来源**：`2509.20279v1.md:229-229`, `2509.20279v1.md:541-549`
  > “TLAgent achieved a mean absolute error (MAE) of 2.047 mm and a root mean square error (RMSE) of 3.091 mm...”
  >
  > “TLAgent consistently delivered the answer with 100% in task completion (success) rate...”
  >
  > “the system achieved a weighted F1-score of 0.939 and an accuracy of 0.931...”

### 6. 正文会主动解释“某些 baseline 为什么表面上看起来还行，但实际上不可靠”

- **风格总结**：这是很像高水平期刊的一点：作者不仅给出结果，还会替读者解释不同指标之间的矛盾，避免审稿人误读。
- **你写时可模仿**：如果你的任务有类不平衡、不同评价指标的偏差，一定要主动解释。
- **原文来源**：`2509.20279v1.md:541-549`, `2509.20279v1.md:1165-1167`
  > “We report Cohen’s κ as the correlation metric because it measures agreement on categorical labels while correcting for chance.”
  >
  > “While some baselines produced comparable F1-scores..., their agreement-based metrics were nearly zero...”

### 7. `Discussion` 不是重复结果，而是先上升到概念层，再落到局限和未来

- **风格总结**：讨论部分一开头就提出 `laboratory intelligence` 这种更高阶概念，然后讲转化价值、可信性、生态意义，最后再谈 limitations 和 future work。
- **你写时可模仿**：Discussion 第一段应该是“这项工作在更大图景中的位置”，不是摘要复读机。
- **原文来源**：`2509.20279v1.md:1459-1469`
  > “We bring the concept of ‘laboratory intelligence’...”
  >
  > “TissueLab bridges the gap between algorithmic innovation and clinical/translational adoption...”
  >
  > “Despite these promising results, our study has several limitations.”

### 8. `Methods` 先讲设计哲学，再讲公式和实现，因此显得既有原则也有技术深度

- **风格总结**：Methods 部分不是直接堆工程细节，而是先解释为什么这样设计，再给形式化表达，再写架构与执行机制。
- **你写时可模仿**：如果你写的是系统型方法，这种顺序会比“直接列模块”更像高水平论文。
- **原文来源**：`2509.20279v1.md:1477-1489`, `2509.20279v1.md:1610-1618`
  > “True evolution requires more than incremental tool updates...”
  >
  > “TissueLab was designed as an adaptive and extensible framework...”
  >
  > “We evaluate TissueLab using a broad spectrum of metrics...”

---

## 三、主文图注风格

### 1. 主文图注的基本格式非常稳定：一句总标题 + `a, b, c...` 面板导航

- **风格总结**：主文 figure caption 的结构非常标准：先一句话概括整图核心，然后按 panel 顺序逐条写明每个面板在展示什么。
- **你写时可模仿**：caption 最主要的任务是让读者快速定位图，而不是替正文重新讲故事。
- **原文来源**：`2509.20279v1.md:215-215`, `2509.20279v1.md:511-511`, `2509.20279v1.md:1359-1359`
  > “Figure 1: Overview of the TissueLab agentic AI ecosystem...”
  >
  > “Figure 2: Agentic AI performs relational analysis with reasoning...”
  >
  > “Figure 5: Clinician-in-the-loop co-evolution improves tissue and cell quantification in cancer research...”

### 2. 主文图注偏“压缩导航型”，每个 panel 用短语或短句，不展开长解释

- **风格总结**：主文图注的 panel 描述往往很短，比如 `Workflow generated...`, `Accuracy comparison...`, `Representative sample...`。核心目标是**高密度导航**。
- **你写时可模仿**：每个 panel 最好先用名词短语或极短句，避免把主文 caption 写成小作文。
- **原文来源**：`2509.20279v1.md:511-511`, `2509.20279v1.md:105-107`
  > “b, Workflow generated by TLAgent...”
  >
  > “c, Correlation with ground truth...”
  >
  > “g, AUC before and after co-evolution with clinician feedback...”

### 3. 主文图注会适度嵌入“解释性尾巴”，让图注顺手承担一点结果解释功能

- **风格总结**：它不只写“图里有什么”，还会轻微点出“这说明了什么”，比如 `illustrating explainable alignment`、`providing explainable correspondence`。
- **你写时可模仿**：可以在 panel 末尾加入半句解释，但别写太满。
- **原文来源**：`2509.20279v1.md:511-511`
  > “f, An example of WSI showing expert annotation and TissueLab prediction overlaid, illustrating explainable alignment.”
  >
  > “l, TissueLab segmentation overlaid on the whole slide images, providing explainable correspondence between predicted metastasis regions and expert labels.”

### 4. 主文图注优先放“主线证据”：workflow、代表图、核心指标、关键对照

- **风格总结**：主文 caption 会优先覆盖最能支撑主结论的面板类型，而不是把所有小验证都塞进去。
- **你写时可模仿**：主文图最好围绕“概念图 + workflow + representative example + quantitative benchmark”组织。
- **原文来源**：`2509.20279v1.md:215-215`, `2509.20279v1.md:511-511`, `2509.20279v1.md:2125-2125`
  > “a, Orchestration phase...”
  >
  > “b, Workflow generated...”
  >
  > “f, An example of WSI...”
  >
  > “c, F1-score comparison...”

---

## 四、补充材料叙述风格

### 1. 补充材料不是再讲一遍主文故事，而是承担“补证据、补界面、补失败模式、补边界条件”的角色

- **风格总结**：从正文对 `Supplementary Figure` 的引用方式来看，补充材料主要用于承接主文中没展开的过程证据，比如界面、active learning 过程、失败案例、混淆矩阵、更多任务细节。
- **你写时可模仿**：补充材料要像“证据仓库”，而不是第二篇主文。
- **原文来源**：`2509.20279v1.md:43-45`, `2509.20279v1.md:223-227`, `2509.20279v1.md:1173-1175`
  > “Once the workflow has been generated (Supplementary Figure 1a)...”
  >
  > “With TissueLab platform... (Supplementary Figure 1b)... (Supplementary Figure 1c)...”
  >
  > “Using TissueLab to automatically generates functional workflows and incorporates clinician annotations in a co-evolving loop (Supplementary Figure 2)...”
  >
  > “As illustrated in Supplementary Figure 4a...”

### 2. 补充材料的叙述重点更偏“过程可见性”和“可验证性”

- **风格总结**：主文里强调结果，补充材料则更强调读者能不能看到中间过程、看到代码检查界面、看到分割结果、看到分类 refinement。它的功能是让主文主张更可信。
- **你写时可模仿**：如果正文已经给出主结论，补充材料就重点放过程可视化、实现流程、额外结果和失败分析。
- **原文来源**：`2509.20279v1.md:1881-1883`, `2509.20279v1.md:2125-2125`
  > “Supplementary Figure 1: TissueLab agentic AI system in use. a, Conversational interface... b, Code inspection interface... c, User interface showing a whole-slide image...”
  >
  > “Supplementary Figure 4: ... workflow construction and visualization to quantitative benchmarking...”

### 3. 补充材料更愿意展示“为什么 aggregate metrics 不够”，因此更强调误差模式和类间表现

- **风格总结**：补充材料会拿 confusion matrix 这类更细颗粒度证据来说明：有些 baseline 看上去总体指标不差，但类别层面仍然错误很多。这是非常典型的 supplementary role。
- **你写时可模仿**：把“额外但关键”的诊断性分析放进 supplement，例如 confusion matrix、error breakdown、failure cases、subgroup results。
- **原文来源**：`2509.20279v1.md:1955-1955`
  > “While some baseline vision-language models appear to achieve reasonable overall metrics, their confusion matrices reveal systematic misclassifications...”
  >
  > “underscoring the importance of evaluating performance not only by aggregate scores but also by class-wise diagnostic fidelity.”

### 4. 补充材料常常负责把主文一句话带过的内容展开成“更完整的技术证据链”

- **风格总结**：例如主文可能只说 `Supplementary Figure 4a` 展示 workflow，但补充里会把 workflow、可视化结果、主任务指标、子任务指标串成一个更完整的验证链条。
- **你写时可模仿**：凡是主文里写成“一句话引用”的部分，都适合在 supplement 中展开成完整说明。
- **原文来源**：`2509.20279v1.md:1173-1179`, `2509.20279v1.md:2125-2125`
  > “TLAgent searched authoritative diagnostic criteria for ICH and assembled a workflow...”
  >
  > “Representative outputs of volumetric segmentation and slice-wise overlays are shown in Supplementary Figure 4b...”
  >
  > “c, F1-score comparison... d, Accuracy comparison... e, F1-score comparison for the fine-grained sub-question...”

---

## 五、补充图注风格

### 1. 补充图注比主文图注更“说明书化”，更像在给额外证据做导览

- **风格总结**：主文图注重压缩；补充图注则更愿意补充上下文和用途，告诉读者这张补图为什么存在、支持哪个主张。
- **你写时可模仿**：Supplementary caption 可以比主文长一点，但核心仍是清楚，不是堆辞藻。
- **原文来源**：`2509.20279v1.md:1881-1883`, `2509.20279v1.md:2125-2125`
  > “Supplementary Figure 1: TissueLab agentic AI system in use...”
  >
  > “Supplementary Figure 4: Automating 3D MRI image analysis for intracranial hemorrhage...”

### 2. 补充图注也保留 `a, b, c...` 面板结构，但描述更完整，常把 panel 的用途直接写出来

- **风格总结**：比如 `a` 不只写 workflow，还会写这是 `using segmentation ... as the basis for diagnosis, followed by code-based quantitative analysis`；也就是说，它不仅说 panel 是什么，还说 panel 在分析链中起什么作用。
- **你写时可模仿**：补图 caption 可以适度加入分析逻辑，而不仅是视觉内容。
- **原文来源**：`2509.20279v1.md:2125-2125`
  > “a, Workflow generated by the agent using segmentation of hemorrhage and brain structures as the basis for diagnosis, followed by code-based quantitative analysis.”

### 3. 补充图注会主动总结整张图的意义，而不只是逐 panel 罗列

- **风格总结**：这是补充图注和主文图注很不同的一点。补充图注末尾常加一个总结句，告诉读者“collectively, these panels show ...”。
- **你写时可模仿**：这非常适合 supplement，因为补图的目的往往就是“进一步证明某件事”。
- **原文来源**：`2509.20279v1.md:2125-2125`
  > “Collectively, these panels show how the TissueLab agentic AI system automates end-to-end 3D MRI analysis... while delivering more consistent and clinically reliable hemorrhage detection than baseline models.”

### 4. 补充图注比主文更愿意放“诊断性指标图”和“失败模式图”

- **风格总结**：例如 confusion matrix 这种更技术化、诊断化的图，在主文里不一定放，但很适合 supplement。caption 也会直接点明它们揭示了哪些系统性错误。
- **你写时可模仿**：如果你有 error map、ablation detail、subgroup confusion、negative case，这类图更适合写进 supplement，并让 caption 明确说出它们揭示了什么。
- **原文来源**：`2509.20279v1.md:1955-1955`
  > “their confusion matrices reveal systematic misclassifications and failure to identify the correct categories.”

---

## 六、如果你现在主要负责“正文 + 图注”，最值得直接套用的写法

### 正文最该学的 5 点

- **开头方式**：先写问题的重要性，再写现有方法的结构性不足。
- **提出方法**：用很干净的转折句推出你的模型/框架。
- **结果段节奏**：任务意义 → 数据/任务 → workflow → 数字 → 对照解释。
- **讨论段功能**：先上升到概念与领域意义，再说 limitations 和 future directions。
- **语言习惯**：多用并列结构，但每个判断都要有数字或机制托底。

### 主文图注最该学的 4 点

- **先给整图总标题**：一句话说清图的中心思想。
- **再按 panel 导航**：`a, b, c...` 逐个写，但保持短。
- **适度加解释性短尾巴**：例如 `illustrating ...`、`showing ...`、`providing ...`。
- **主文只放主线证据**：workflow、代表图、核心指标、关键对照。

### 补充材料最该学的 4 点

- **别重讲主文故事**：补充材料是证据仓库，不是第二篇主文。
- **多放过程与边界证据**：界面、流程、失败模式、类间差异、额外 benchmark。
- **caption 可以更完整**：允许比主文多写一点逻辑。
- **补图末尾可加总结句**：尤其适合说明“这组面板共同证明了什么”。

---

## 七、可直接迁移的句型模板

### 正文模板

- **问题开场**：`X is critical for ..., yet existing methods remain limited by ...`
- **提出方案**：`To address these challenges, we introduce ...`
- **结果汇报**：`As a result, ... achieved ..., outperforming ...`
- **意义解释**：`These findings highlight ...` / `These results demonstrate that ...`
- **讨论拔高**：`Together, these findings position ... as ...`

### 主文图注模板

- **总标题句**：`Figure X: [一句话概括整图核心信息].`
- **面板导航句**：`a, ... . b, ... . c, ... .`
- **轻解释尾巴**：`..., illustrating ...` / `..., providing ...`

### 补充图注模板

- **总标题句**：`Supplementary Figure X: [补充验证主题].`
- **面板说明句**：`a, ... as the basis for ... . b, ... . c, ... .`
- **整图总结句**：`Collectively, these panels show that ...`

---

## 八、对你接下来写作的实际建议

如果你要写 **Nature 子刊风格**，最实用的策略不是“把全文都写成一种腔调”，而是明确区分：

- **正文**：讲价值、讲主线、讲最强证据。
- **主文图注**：做高密度导航。
- **补充材料**：做证据延展、过程展开、边界验证。
- **补充图注**：把额外验证解释成“为什么主文可信”。

你现在最应该优先模仿的是：**正文结果段的推进节奏** 和 **主文图注的 panel 组织方式**。如果后面你要写 supplement，我建议再按这份四分版单独起草，不要直接把主文写法平移过去。

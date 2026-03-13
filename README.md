# small-LLM
my implemenntation of happy-LLM

## transformer -- 已完成
目标：搭建起一个完整的 Transformer 模型。

2026-03-12 v1.0 在happy-LLM的基础上，基于pytorch实现了位置编码、多头注意力、层归一化、前馈神经网络、编码器、解码器等模块，并优化了部分代码。通过train_smoke.py脚本进行测试结果显示模型可正常工作。

可以改进的地方：
1. 注意力机制（以及多头注意力）进一步优化性能
2. 手动实现embedding层（分词器）


## 预训练语言模型 -- 未完成
承接上一部分，transformer架构是一种encoder-decoder的架构。编码器解码器具有不同的结构。针对 Encoder、Decoder 的特点，引入 ELMo(embedding from language model)的预训练思路，开始出现不同的、对 Transformer 进行优化的思路。
> 例如，Google 仅选择了 Encoder 层，通过将 Encoder 层进行堆叠，再提出不同的预训练任务-掩码语言模型（Masked Language Model，MLM），打造了一统自然语言理解（Natural Language Understanding，NLU）任务的代表模型——BERT。而 OpenAI 则选择了 Decoder 层，使用原有的语言模型（Language Model，LM）任务，通过不断增加模型参数和预训练语料，打造了在 NLG（Natural Language Generation，自然语言生成）任务上优势明显的 GPT 系列模型，也是现今大火的 LLM 的基座模型。当然，还有一种思路是同时保留 Encoder 与 Decoder，打造预训练的 Transformer 模型，例如由 Google 发布的 T5模型。

### Encoder-only PLM
Bert的预训练任务创新——MLM Mask language model和Next sentence prediction。MLM即为模拟完形填空，随机遮蔽一部分token，让模型根据未遮蔽的部分预测被遮蔽的token。

NSP即为下一个句子预测。核心任务是让模型判断一个句对的两个句子是否是连续上下文。
> NSP 的核心思想是针对句级的 NLU 任务，例如问答匹配、自然语言推理等。问答匹配是指，输入一个问题和若干个回答，要求模型找出问题的真正回答；自然语言推理是指，输入一个前提和一个推理，判断推理是否是符合前提的。

确立了预训练-微调的两阶段思想，即在海量无监督语料上进行预训练来获得通用的文本理解与生成能力，再在对应的下游任务上进行微调。该种思想的一个重点在于，预训练得到的强大能力能否通过低成本的微调快速迁移到对应的下游任务上。

主要是bert和bert的一些变种吧。如roberta就是采取了更大的参数规模和在预训练中取消了nsp；然后albert探索了一些提升模型宽度同时降低模型参数的手段，事实证明提升模型宽度可以提升模型的表现，是合理的技术路径。还有个很有价值的探索是它尝试改进nsp（不仅要预测是否正例还要预测是否顺序）并取得了很好的效果。
### Encoder-Decoder PLM
主要是T5模型。用了一个归一化采用一个新的函数，均方根函数 RMSNorm，只有一个放缩因子。


### Decoder-only PLM
Decoder-Only 的模型结构往往更适合于文本生成任务，因此，Decoder-Only 模型往往选择了最传统也最直接的预训练任务——因果语言模型，Causal Language Model，下简称 CLM。 

Decoder-Only 就是目前大火的 LLM 的基础架构，目前所有的 LLM 基本都是 Decoder-Only 模型（RWKV、Mamba 等非 Transformer 架构除外）。

> GPT-2 的另一个重大突破是以 zero-shot（零样本学习）为主要目标，也就是不对模型进行微调，直接要求模型解决任务。例如，在传统的预训练-微调范式中，我们要解决一个问题，一般需要收集几百上千的训练样本，在这些训练样本上微调预训练语言模型来实现该问题的解决。而 zero-shot 则强调不使用任何训练样本，直接通过向预训练语言模型描述问题来去解决该问题。zero-shot 的思路自然是比预训练-微调范式更进一步、更高效的自然语言范式，但是在 GPT-2 的时代，模型能力还不足够支撑较好的 zero-shot 效果，在大模型时代，zero-shot 及其延伸出的 few-shot（少样本学习）才开始逐渐成为主流。

开源比较火的有LLaMA和GLM，也许Qwen也是？

## 大语言模型 
具有以下特点：
1. Large language model 指的是上百亿或者更多更多参数的语言模型，广义上十几亿（Qwen1.5B）或者上千亿的都可以说是LLM。最大的特点是涌现能力（在同样的模型架构和预训练任务下，在小模型的时候能力不突出，模型规模和token语料上去后，性能超过传统语言模型（bert等）），量变引起了质变。

2. 上下文学习 上下文学习能力是由 GPT-3 首次引入的。具体而言，上下文学习是指允许语言模型在提供自然语言指令或多个任务示例的情况下，通过理解上下文并生成相应输出的方式来执行任务，而无需额外的训练或参数更新。无需微调，无需算力。这个是few-shot
3. 指令遵循。zero-shot。无需给出理解，给出指令让模型工作。这也是通用模型的伟大之处，使得一个大模型可以用在各行各业。
4. 逐步推理。LLM 通过采用思维链（Chain-of-Thought，CoT）推理策略，可以利用包含中间推理步骤的提示机制来解决复杂任务
5. 多语言支持，因为训练语料包含多国语言。长文本理解，分布式集群训练+具有外推能力的位置编码；拓展多模态；幻觉，可通过prompt+RAG缓解。

### 如何训练一个LLM

### 预训练 pretrain
它们的预训练任务也都沿承了 GPT 模型的经典预训练任务——因果语言模型（Causal Language Model，CLM）。而且具有夸张的模型参数量以及训练语料，高达上千亿参数（GPT3 175B参数 300B数据量）。




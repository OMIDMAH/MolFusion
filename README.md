# MolFusion

### A Modular Multi-Agent Molecular Feature Extraction Framework

<p align="center">
  <img src="https://github.com/OMIDMAH/MolFusion/blob/main/MolFusion_edited.png" width="1000"/>
</p>

MolFusion is a fully modular molecular representation learning pipeline integrating 12 independent feature extraction agents for cheminformatics, QSAR, ADMET prediction, molecular generation, and deep learning workflows.

## 🌐 Live Application
You can use the fully operational web app directly here:
👉 **[Launch MolFusion Live Web Suite](https://omidmah.github.io/MolFusion/)**

The framework combines:

- Agent 1: Morgan ECFP (r=2, 1024 bits)
- Agent 2: Avalon (1024 bits)
- Agent 3: ErG (315 bits)
- Agent 4: MACCS keys (167 bits)
- Agent 5: RDKit Physicochemical (217 descriptors)
- Agent 6: SMILES TF-IDF (32 dimensions)
- Agent 7: SELFIES Tokenization
- Agent 8: Group SELFIES Fixed-Length
- Agent 9: GNN Embeddings (128 dimensions)
- Agent 10: VAE Latent Vectors (128 dimensions)
- Agent 11: Fragment/Substructure Patterns (17 features)
- Agent 12: 3D Shape Descriptors (9 features)

**Producer:** OMID M.

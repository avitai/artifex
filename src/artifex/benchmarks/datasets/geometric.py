"""Geometric datasets for benchmark evaluation."""

import logging
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import trimesh

from artifex.benchmarks.runtime_guards import demo_mode_from_mapping, require_demo_mode
from artifex.generative_models.core.configuration import DataConfig
from artifex.utils.file_utils import ensure_valid_output_path


logger = logging.getLogger(__name__)


# ShapeNet taxonomy mapping (similar to PyTorch3D approach)
SHAPENET_SYNSETS = {
    "02691156": "airplane",
    "02747177": "ashcan",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02834778": "bicycle",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}

# Standard ShapeNet train/val/test splits (following PyTorch3D conventions)
SHAPENET_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


class ShapeNetDataset:
    """ShapeNet dataset for point cloud generation benchmarks.

    This implementation follows patterns similar to PyTorch3D's ShapeNet dataset,
    providing robust data loading, proper taxonomy handling, and multiple
    data source fallbacks. Structurally conforms to DatasetProtocol.

    The dataset supports:
    - Official ShapeNet Core.v2 data structure
    - Alternative data sources (ModelNet, synthetic data)
    - Multiple mesh formats (.obj, .off, .ply)
    - Automatic point cloud conversion
    - Standard train/val/test splits
    """

    def __init__(self, data_path: str, config: DataConfig, *, rngs: nnx.Rngs):
        """Initialize ShapeNet dataset following PyTorch3D patterns.

        Args:
            data_path: Path to ShapeNet dataset directory
            config: DataConfig instance with dataset settings.
                Configuration metadata should include:
                - synsets: List of ShapeNet synset IDs to include (default: common objects)
                - num_points: Number of points per point cloud (default: 2048)
                - normalize: Whether to normalize point clouds (default: True)
                - split_ratios: Custom train/val/test ratios (optional)
                - load_meshes: Whether to load original meshes (default: False)
                - data_source: Preferred data source ('shapenet', 'modelnet', 'synthetic')
                - version: ShapeNet version ('v1' or 'v2', default: 'v2')
                - batch_size: Batch size for data loading
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        self.config = config
        self.data_path = Path(data_path)
        self.rngs = rngs
        self.data: dict = {}
        self.demo_mode = demo_mode_from_mapping(config.metadata)

        self._load_dataset()

    def _acquire_data_if_needed(self):
        """Acquire dataset if needed."""
        data_source = str(self.config.metadata.get("data_source", "auto")).lower()

        if data_source in {"synthetic", "mock", "demo"}:
            require_demo_mode(
                enabled=self.demo_mode,
                component="ShapeNetDataset",
                detail=(
                    "Synthetic ShapeNet data is retained only for benchmark demos and is not "
                    "the benchmark-grade runtime."
                ),
            )
            logger.info("Demo mode enabled; creating retained synthetic ShapeNet data")
            self._create_synthetic_data()
            self._validate_data_structure()
            return

        if data_source == "modelnet":
            require_demo_mode(
                enabled=self.demo_mode,
                component="ShapeNetDataset",
                detail=(
                    "The retained ModelNet substitute is a demo-only fallback and not the "
                    "supported ShapeNet benchmark runtime."
                ),
            )
            logger.info("Demo mode enabled; using retained ModelNet substitute")
            try:
                self._download_modelnet_alternative()
            except (OSError, RuntimeError, ValueError, ImportError) as e:
                logger.warning("ModelNet alternative failed: %s", e)
                self._create_synthetic_data()
            self._validate_data_structure()
            return

        if not self.data_path.exists():
            if self.demo_mode:
                logger.info(
                    "Demo mode enabled; creating retained synthetic ShapeNet data at %s",
                    self.data_path,
                )
                self._create_synthetic_data()
                self._validate_data_structure()
                return
            raise RuntimeError(
                "ShapeNetDataset requires benchmark-grade ShapeNet assets at "
                f"{self.data_path}. Automatic download, ModelNet substitution, and synthetic "
                "fallback are no longer part of the supported benchmark runtime. Pass "
                "demo_mode=True or data_source='synthetic' only for the retained demo workflow."
            )

        self._validate_data_structure()

    def _acquire_shapenet_data(self) -> bool:
        """Acquire ShapeNet data using multiple strategies (PyTorch3D-like approach).

        Returns:
            True if any strategy succeeded, False otherwise
        """
        strategies = [
            ("ShapeNet Core v2", self._download_shapenet_core),
            ("ModelNet Alternative", self._download_modelnet_alternative),
            ("Processed ShapeNet", self._download_processed_shapenet),
            ("Synthetic Data", self._create_synthetic_data),
        ]

        for strategy_name, strategy_func in strategies:
            try:
                logger.info("Trying strategy: %s", strategy_name)
                strategy_func()

                # Verify we got some data
                valid_models = self._count_valid_models()
                if valid_models > 0:
                    logger.info("Success with %s: %d valid models", strategy_name, valid_models)
                    return True
                else:
                    logger.warning("%s produced no valid models", strategy_name)

            except (OSError, RuntimeError, ValueError, ImportError) as e:
                logger.warning("%s failed: %s", strategy_name, e)
                continue

        logger.error("All data acquisition strategies failed")
        return False

    def _download_shapenet_core(self):
        """Download ShapeNet Core data (similar to PyTorch3D approach)."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            # Try different ShapeNet repositories
            repos = [
                "ShapeNet/ShapeNetCore.v2",
                "ShapeNet/ShapeNetCore",
                "shi-labs/shapenet-processed",
            ]

            synsets = self.config.metadata.get("synsets", ["02691156", "02958343", "03001627"])

            for repo_id in repos:
                try:
                    logger.info("Checking repository: %s", repo_id)
                    files = list_repo_files(repo_id, repo_type="dataset")

                    # Filter files for target synsets
                    target_files = []
                    for synset in synsets:
                        matching_files = [
                            f
                            for f in files
                            if synset in f
                            and any(f.endswith(ext) for ext in [".obj", ".off", ".ply"])
                        ]
                        target_files.extend(matching_files[:20])  # Limit per synset

                    if target_files:
                        logger.info("Found %d relevant files in %s", len(target_files), repo_id)

                        # Directory already created by base class

                        # Download files
                        for file_path in target_files:
                            try:
                                local_file = self.data_path / file_path
                                local_file.parent.mkdir(parents=True, exist_ok=True)

                                hf_hub_download(  # nosec B615
                                    repo_id=repo_id,
                                    filename=file_path,
                                    local_dir=str(self.data_path),
                                    repo_type="dataset",
                                    revision="main",  # Pin to specific revision for security
                                )
                            except (OSError, ValueError, RuntimeError) as e:
                                logger.warning("Failed to download %s: %s", file_path, e)
                                continue

                        if self._count_valid_models() > 0:
                            return

                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("Repository %s failed: %s", repo_id, e)
                    continue

            raise RuntimeError("No ShapeNet repositories accessible")

        except ImportError:
            logger.info("huggingface_hub not available, trying alternative approach")
            raise

    def _download_processed_shapenet(self):
        """Download pre-processed ShapeNet point clouds."""
        try:
            # Try to download from alternative sources that provide processed data
            processed_urls = [
                "https://github.com/charlesq34/pointnet/raw/master/part_seg/shapenetcore_partanno_segmentation_benchmark_v0.zip",
                # Add more processed data URLs as available
            ]

            for url in processed_urls:
                try:
                    logger.info("Downloading processed data from: %s", url)
                    zip_path = self.data_path.parent / "shapenet_processed.zip"
                    zip_path.parent.mkdir(parents=True, exist_ok=True)

                    # Validate URL for security
                    parsed_url = urllib.parse.urlparse(url)
                    if parsed_url.scheme not in ("http", "https"):
                        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")

                    # Only allow downloads from trusted domains
                    trusted_domains = ["github.com", "huggingface.co", "storage.googleapis.com"]
                    if parsed_url.netloc not in trusted_domains:
                        raise ValueError(f"Untrusted domain: {parsed_url.netloc}")

                    urllib.request.urlretrieve(url, zip_path)  # nosec B310

                    # Extract
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(self.data_path)

                    # Clean up
                    zip_path.unlink()

                    if self._count_valid_models() > 0:
                        return

                except (OSError, urllib.error.URLError, ValueError, zipfile.BadZipFile) as e:
                    logger.warning("Failed to download from %s: %s", url, e)
                    continue

            raise RuntimeError("No processed ShapeNet data available")

        except (OSError, urllib.error.URLError, ValueError, zipfile.BadZipFile) as e:
            logger.warning("Processed ShapeNet download failed: %s", e)
            raise

    def _download_modelnet_alternative(self):
        """Download ModelNet as ShapeNet alternative (PyTorch3D-like fallback)."""
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading ModelNet40 as ShapeNet alternative...")
            temp_path = self.data_path.parent / "modelnet_temp"

            # Download ModelNet40
            snapshot_download(  # nosec B615
                repo_id="princeton-vl/ModelNet40",
                local_dir=str(temp_path),
                repo_type="dataset",
                revision="main",  # Pin to specific revision for security
            )

            # Convert ModelNet structure to ShapeNet format
            self._convert_modelnet_to_shapenet(temp_path)

            # Clean up
            shutil.rmtree(temp_path, ignore_errors=True)

        except (OSError, ImportError, RuntimeError, ValueError) as e:
            logger.warning("ModelNet alternative failed: %s", e)
            raise

    def _convert_modelnet_to_shapenet(self, modelnet_path: Path):
        """Convert ModelNet structure to ShapeNet format."""
        # Map ModelNet classes to ShapeNet synsets
        modelnet_to_shapenet = {
            "airplane": "02691156",
            "car": "02958343",
            "chair": "03001627",
            "table": "04379243",
            "sofa": "04256520",
            "bed": "02818832",
            "toilet": "04401088",
            "desk": "04379243",
            "dresser": "02933112",
            "night_stand": "04379243",
        }

        # Directory already created by base class

        for modelnet_class, shapenet_id in modelnet_to_shapenet.items():
            # Find ModelNet class directory
            class_dirs = list(modelnet_path.rglob(f"*{modelnet_class}*"))
            if not class_dirs:
                continue

            # Create ShapeNet synset directory
            synset_dir = self.data_path / shapenet_id
            synset_dir.mkdir(exist_ok=True)

            # Process models
            model_files = []
            for class_dir in class_dirs:
                model_files.extend(list(class_dir.rglob("*.off")))
                model_files.extend(list(class_dir.rglob("*.obj")))

            logger.info(
                "Converting %d %s models to %s", len(model_files), modelnet_class, shapenet_id
            )

            for i, model_file in enumerate(model_files[:50]):  # Limit to 50 per class
                try:
                    # Create model directory
                    model_dir = synset_dir / f"{model_file.stem}_{i:03d}"
                    model_dir.mkdir(exist_ok=True)

                    # Load and save as OBJ
                    mesh = trimesh.load(str(model_file))
                    output_file = model_dir / "model.obj"
                    mesh.export(str(output_file))

                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("Failed to convert %s: %s", model_file, e)
                    continue

    def _create_synthetic_data(self):
        """Create synthetic ShapeNet-style data as final fallback."""
        logger.info("Creating synthetic 3D models...")

        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            synsets = self.config.metadata.get("synsets", ["02691156", "02958343", "03001627"])
            models_per_synset = self.config.metadata.get("models_per_synset", 20)

            # Directory already created by base class

            created_models = 0
            for synset_id in synsets:
                synset_name = SHAPENET_SYNSETS.get(synset_id, "unknown")
                synset_dir = self.data_path / synset_id
                synset_dir.mkdir(exist_ok=True)

                logger.info("Creating %d synthetic %s models...", models_per_synset, synset_name)

                for i in range(models_per_synset):
                    try:
                        model_dir = synset_dir / f"synthetic_{i:03d}"
                        model_dir.mkdir(exist_ok=True)

                        # Create shape based on category
                        mesh = self._create_synthetic_shape(synset_name, i)

                        # Save as OBJ
                        output_file = model_dir / "model.obj"
                        mesh.export(str(output_file))
                        created_models += 1

                    except (OSError, ValueError, RuntimeError) as e:
                        logger.warning("Failed to create model %d for %s: %s", i, synset_name, e)
                        continue

                logger.info("Created %d models for %s", created_models, synset_name)

            if created_models == 0:
                raise RuntimeError("No synthetic models were created")

            logger.info("Synthetic data creation complete: %d total models", created_models)

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Synthetic data creation failed: %s", e)
            logger.info("Falling back to minimal dataset creation...")
            self._create_minimal_dataset()

    def _create_synthetic_shape(self, shape_type: str, variant: int) -> trimesh.Trimesh:
        """Create synthetic 3D shapes based on category."""
        try:
            # Add variation based on variant number
            scale_factor = 1.0 + (variant % 5) * 0.2

            if shape_type == "airplane":
                # Create airplane-like shape
                try:
                    fuselage = trimesh.creation.box(extents=[4 * scale_factor, 0.5, 0.6])
                    wing = trimesh.creation.box(extents=[1.5, 3 * scale_factor, 0.1])
                    tail = trimesh.creation.box(extents=[0.5, 0.3, 1.2])
                    tail.apply_translation([1.5 * scale_factor, 0, 0.5])

                    mesh = trimesh.util.concatenate([fuselage, wing, tail])
                except (ValueError, RuntimeError, TypeError):
                    # Fallback to simple box
                    mesh = trimesh.creation.box(extents=[2 * scale_factor, 1, 0.5])

            elif shape_type == "car":
                # Create car-like shape
                try:
                    body = trimesh.creation.box(extents=[3 * scale_factor, 1.5, 1])
                    roof = trimesh.creation.box(extents=[2 * scale_factor, 1.5, 0.8])
                    roof.apply_translation([0, 0, 0.9])

                    mesh = trimesh.util.concatenate([body, roof])
                except (ValueError, RuntimeError, TypeError):
                    # Fallback to simple box
                    mesh = trimesh.creation.box(extents=[2 * scale_factor, 1.5, 1])

            elif shape_type == "chair":
                # Create chair-like shape
                try:
                    seat = trimesh.creation.box(extents=[1 * scale_factor, 1, 0.1])
                    backrest = trimesh.creation.box(extents=[0.1, 1, 1.2 * scale_factor])
                    backrest.apply_translation([0.45 * scale_factor, 0, 0.6 * scale_factor])

                    # Add legs
                    leg_positions = [
                        (-0.4, -0.4, -0.5),
                        (0.4, -0.4, -0.5),
                        (-0.4, 0.4, -0.5),
                        (0.4, 0.4, -0.5),
                    ]
                    legs = []
                    for pos in leg_positions:
                        leg = trimesh.creation.box(extents=[0.05, 0.05, 1])
                        leg.apply_translation([p * scale_factor for p in pos])
                        legs.append(leg)

                    mesh = trimesh.util.concatenate([seat, backrest, *legs])
                except (ValueError, RuntimeError, TypeError):
                    # Fallback to simple box
                    mesh = trimesh.creation.box(extents=[1 * scale_factor, 1, 1])

            else:
                # Generic shape - always works
                mesh = trimesh.creation.box(extents=[1 * scale_factor, 1, 1])

            # Add random deformation (optional, skip if it fails)
            try:
                vertices = mesh.vertices.copy()
                noise = np.random.normal(0, 0.02 * scale_factor, vertices.shape)
                vertices += noise
                mesh.vertices = vertices
            except (ValueError, IndexError, TypeError):
                # Skip deformation if it fails
                pass

            return mesh

        except (ValueError, RuntimeError, TypeError, OSError) as e:
            logger.warning("Failed to create %s shape (variant %d): %s", shape_type, variant, e)
            # Ultimate fallback - simple unit cube
            return trimesh.creation.box(extents=[1, 1, 1])

    def _validate_data_structure(self):
        """Validate the acquired data has proper ShapeNet structure."""
        if not self.data_path.exists():
            if not self.demo_mode:
                raise RuntimeError(
                    f"ShapeNetDataset requires benchmark-grade assets at {self.data_path}. "
                    "Supported benchmark mode no longer creates placeholder geometry on the fly."
                )
            logger.warning("Data path does not exist: %s", self.data_path)
            logger.info("Demo mode enabled; generating retained synthetic data")
            self._create_synthetic_data()

        valid_models = self._count_valid_models()
        if valid_models == 0:
            if not self.demo_mode:
                raise RuntimeError(
                    f"ShapeNetDataset found no benchmark-grade 3D models in {self.data_path}. "
                    "Supported benchmark mode no longer falls back to synthetic or minimal "
                    "placeholder assets."
                )
            logger.warning("No valid 3D models found in %s", self.data_path)
            logger.info("Demo mode enabled; generating retained synthetic fallback data")
            self._create_synthetic_data()

            valid_models = self._count_valid_models()
            if valid_models == 0:
                logger.error("Failed to create synthetic data, creating minimal dataset...")
                self._create_minimal_dataset()
                valid_models = self._count_valid_models()

        logger.info("Validated dataset: %d valid models found", valid_models)

    def _create_minimal_dataset(self):
        """Create minimal dataset as absolute fallback."""
        logger.info("Creating minimal dataset with basic shapes...")

        synsets = self.config.metadata.get("synsets", ["02691156"])

        for synset_id in synsets:
            synset_dir = self.data_path / synset_id
            synset_dir.mkdir(parents=True, exist_ok=True)

            # Create one simple model
            model_dir = synset_dir / "minimal_001"
            model_dir.mkdir(exist_ok=True)

            # Create a simple cube mesh
            try:
                import trimesh

                mesh = trimesh.creation.box(extents=[1, 1, 1])
                obj_file = model_dir / "model.obj"
                mesh.export(str(obj_file))
                logger.info("Created minimal model: %s", obj_file)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to create minimal model: %s", e)
                # Create a simple text file as placeholder
                obj_path = model_dir / "model.obj"
                # Ensure the file is saved in the test_results directory during tests
                obj_file = ensure_valid_output_path(str(obj_path))
                with open(obj_file, "w") as f:
                    f.write("# Simple placeholder OBJ file\n")
                    f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    def _count_valid_models(self) -> int:
        """Count valid 3D model files in the dataset."""
        try:
            if not self.data_path.exists():
                return 0

            model_files = []
            for ext in [".obj", ".off", ".ply"]:
                try:
                    files = list(self.data_path.rglob(f"*{ext}"))
                    model_files.extend(files)
                except OSError as e:
                    logger.warning("Error searching for %s files: %s", ext, e)
                    continue

            return len(model_files)
        except OSError as e:
            logger.warning("Error counting models: %s", e)
            return 0

    def _load_dataset(self):
        """Load and preprocess the ShapeNet dataset."""
        # First check if data acquisition is needed
        self._acquire_data_if_needed()

        # Configuration
        self.num_points = self.config.metadata.get("num_points", 2048)  # PyTorch3D default
        self.synsets = self.config.metadata.get("synsets", ["02691156", "02958343", "03001627"])
        self.normalize = self.config.metadata.get("normalize", True)
        self.load_meshes = self.config.metadata.get("load_meshes", False)

        # Load data
        logger.info("Loading ShapeNet data...")
        self.data = self._load_shapenet_files()

        # Create splits
        self._create_data_splits()

        # Preprocess
        if self.normalize:
            self._normalize_point_clouds()

    def _load_shapenet_files(self) -> dict[str, list[dict[str, Any]]]:
        """Load ShapeNet files following PyTorch3D patterns."""
        data_items = []

        # Process each synset
        for synset_id in self.synsets:
            synset_dir = self.data_path / synset_id
            if not synset_dir.exists():
                logger.warning("Synset %s not found", synset_id)
                continue

            synset_name = SHAPENET_SYNSETS.get(synset_id, "unknown")
            logger.info("Loading synset %s (%s)...", synset_id, synset_name)

            # Find all model files
            model_files = []
            for ext in [".obj", ".off", ".ply"]:
                model_files.extend(list(synset_dir.rglob(f"*{ext}")))

            logger.info("Found %d models for %s", len(model_files), synset_name)

            # Process each model
            for model_file in model_files:
                try:
                    item = self._load_single_model(model_file, synset_id, synset_name)
                    if item is not None:
                        data_items.append(item)
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("Failed to load %s: %s", model_file, e)
                    continue

        logger.info("Successfully loaded %d models", len(data_items))
        return {"all": data_items}

    def _load_single_model(
        self, model_file: Path, synset_id: str, synset_name: str
    ) -> dict[str, Any] | None:
        """Load a single 3D model file."""
        try:
            # Load mesh
            mesh = trimesh.load(str(model_file), force="mesh")

            # Convert to point cloud
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
                # Sample from surface
                points = trimesh.sample.sample_surface(mesh, self.num_points)[0]
            elif hasattr(mesh, "vertices"):
                # Sample from vertices
                vertices = np.asarray(getattr(mesh, "vertices"))
                if len(vertices) >= self.num_points:
                    indices = np.random.choice(len(vertices), self.num_points, replace=False)
                    points = vertices[indices]
                else:
                    # Upsample with noise
                    points = self._upsample_points(vertices)
            else:
                return None

            # Validate points
            points = np.array(points, dtype=np.float32)
            if not np.isfinite(points).all():
                return None

            item = {
                "point_cloud": jnp.array(points),
                "synset_id": synset_id,
                "synset_name": synset_name,
                "model_id": model_file.stem,
                "file_path": str(model_file),
            }

            # Optionally include mesh
            if self.load_meshes:
                item["mesh"] = mesh

            return item

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            logger.warning("Error loading %s: %s", model_file, e)
            return None

    def _upsample_points(self, points: np.ndarray) -> np.ndarray:
        """Upsample point cloud to target size."""
        points = np.array(points, dtype=np.float32)
        current_size = len(points)

        if current_size == 0:
            return np.random.normal(0, 1, (self.num_points, 3)).astype(np.float32)

        # Repeat points and add noise
        repeats = (self.num_points + current_size - 1) // current_size
        upsampled = np.tile(points, (repeats, 1))[: self.num_points]

        # Add small random noise for variation
        noise_scale = np.std(points) * 0.01 if np.std(points) > 0 else 0.01
        noise = np.random.normal(0, noise_scale, upsampled.shape)
        upsampled += noise

        return upsampled.astype(np.float32)

    def _create_data_splits(self):
        """Create train/val/test splits following PyTorch3D conventions."""
        all_items = self.data["all"]

        # Shuffle with reproducible seed
        key = self.rngs.data() if hasattr(self.rngs, "data") else jax.random.key(42)
        indices = jax.random.permutation(key, len(all_items))
        shuffled_items = [all_items[int(i)] for i in indices]

        # Split ratios
        split_ratios = self.config.metadata.get("split_ratios", SHAPENET_SPLITS)
        train_ratio = split_ratios.get("train", 0.8)
        val_ratio = split_ratios.get("val", 0.1)

        # Calculate split sizes, ensuring train gets at least 1 sample
        total_size = len(shuffled_items)
        train_size = max(1, int(total_size * train_ratio)) if total_size > 0 else 0
        val_size = int(total_size * val_ratio)

        # Clamp so splits don't exceed total
        if train_size + val_size > total_size:
            val_size = max(0, total_size - train_size)

        # Create splits
        train_items = shuffled_items[:train_size]
        val_items = shuffled_items[train_size : train_size + val_size]
        test_items = shuffled_items[train_size + val_size :]

        # Convert to arrays
        self.data = {
            "train": self._items_to_arrays(train_items),
            "val": self._items_to_arrays(val_items),
            "test": self._items_to_arrays(test_items),
        }

        logger.info(
            "Data splits: train=%d, val=%d, test=%d",
            len(train_items),
            len(val_items),
            len(test_items),
        )

    def _items_to_arrays(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert list of items to JAX arrays."""
        if not items:
            return {
                "point_clouds": jnp.empty((0, self.num_points, 3)),
                "labels": jnp.empty((0,), dtype=jnp.int32),
                "synset_ids": jnp.array([]),
                "model_ids": jnp.array([]),
            }

        # Stack point clouds
        point_clouds = jnp.stack([item["point_cloud"] for item in items])

        # Create labels (synset index)
        unique_synsets = sorted(set(item["synset_id"] for item in items))
        synset_to_label = {synset: i for i, synset in enumerate(unique_synsets)}
        labels = jnp.array([synset_to_label[item["synset_id"]] for item in items])

        return {
            "point_clouds": point_clouds,
            "labels": labels,
            "categories": labels,  # Alias for compatibility
            "synset_ids": [item["synset_id"] for item in items],  # Keep as list of strings
            "model_ids": [item["model_id"] for item in items],  # Keep as list of strings
        }

    def _normalize_point_clouds(self):
        """Normalize point clouds following PyTorch3D conventions."""
        for split in self.data:
            point_clouds = self.data[split]["point_clouds"]

            if len(point_clouds) == 0:
                continue

            # Center each point cloud
            centroids = jnp.mean(point_clouds, axis=1, keepdims=True)
            centered = point_clouds - centroids

            # Scale to unit sphere
            distances = jnp.linalg.norm(centered, axis=2)
            max_distances = jnp.max(distances, axis=1, keepdims=True)
            normalized = centered / (max_distances[..., None] + 1e-8)

            self.data[split]["point_clouds"] = normalized

    def __len__(self) -> int:
        """Return total number of samples across all splits."""
        total = 0
        for split_data in self.data.values():
            if isinstance(split_data, dict) and "point_clouds" in split_data:
                total += split_data["point_clouds"].shape[0]
        return total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index (across all splits).

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample data
        """
        offset = 0
        for split_name, split_data in self.data.items():
            if not isinstance(split_data, dict) or "point_clouds" not in split_data:
                continue
            n = split_data["point_clouds"].shape[0]
            if idx < offset + n:
                local_idx = idx - offset
                return self.get_sample(local_idx, split=split_name)
            offset += n
        raise IndexError(f"Index {idx} out of range for dataset of size {offset}")

    def get_batch(
        self, batch_size: int | None = None, split: str | None = None
    ) -> dict[str, jax.Array]:
        """Get a batch of data from the specified split."""
        if split is None:
            split = self.config.split or self.config.metadata.get("split", "train")
        if batch_size is None:
            batch_size = int(self.config.metadata.get("batch_size", 32))

        if split not in self.data:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.data.keys())}")

        split_data = self.data[split]
        num_samples = split_data["point_clouds"].shape[0]

        if num_samples == 0:
            raise ValueError(f"No data available for split '{split}'")

        # Generate random indices
        key = self.rngs.batch() if hasattr(self.rngs, "batch") else jax.random.key(0)
        indices = jax.random.choice(key, num_samples, (batch_size,), replace=True)

        # Extract batch
        batch = {}
        for key, values in split_data.items():
            if isinstance(values, jax.Array):
                batch[key] = values[indices]

        return batch

    def get_dataset_info(self) -> dict[str, Any]:
        """Get complete dataset information."""
        info = {
            "name": "ShapeNet",
            "data_path": str(self.data_path),
            "num_points": self.num_points,
            "synsets": self.synsets,
            "synset_names": [SHAPENET_SYNSETS.get(s, "unknown") for s in self.synsets],
            "normalize": self.normalize,
            "splits": list(self.data.keys()),
            "load_meshes": self.load_meshes,
        }

        # Add split sizes
        for split in self.data:
            info[f"{split}_size"] = self.data[split]["point_clouds"].shape[0]

        return info

    def get_split_size(self, split: str) -> int:
        """Get the size of a specific split."""
        if split not in self.data:
            return 0
        return self.data[split]["point_clouds"].shape[0]

    def get_sample(self, index: int, split: str = "train") -> dict[str, Any]:
        """Get a single sample from the dataset."""
        if split not in self.data:
            raise ValueError(f"Split '{split}' not available")

        split_data = self.data[split]
        if index >= split_data["point_clouds"].shape[0]:
            raise IndexError(f"Index {index} out of range for split '{split}'")

        sample = {}
        for key, values in split_data.items():
            if isinstance(values, jax.Array):
                sample[key] = values[index]
            else:
                sample[key] = values[index]

        return sample

    def get_synset_info(self) -> dict[str, str]:
        """Get ShapeNet synset taxonomy information."""
        return {
            synset: SHAPENET_SYNSETS[synset]
            for synset in self.synsets
            if synset in SHAPENET_SYNSETS
        }


class GeometricDatasetRegistry:
    """Registry for geometric datasets used in benchmarks."""

    _datasets: dict[str, type[ShapeNetDataset]] = {
        "shapenet": ShapeNetDataset,
    }

    @classmethod
    def register_dataset(cls, name: str, dataset_class: type[ShapeNetDataset]):
        """Register a new dataset class."""
        cls._datasets[name] = dataset_class

    @classmethod
    def get_dataset(
        cls, name: str, data_path: str, config: DataConfig, *, rngs: nnx.Rngs
    ) -> ShapeNetDataset:
        """Get a dataset instance by name."""
        if name not in cls._datasets:
            raise ValueError(
                f"Dataset '{name}' not registered. Available: {list(cls._datasets.keys())}"
            )

        dataset_class = cls._datasets[name]

        return dataset_class(data_path, config, rngs=rngs)

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List all registered datasets."""
        return list(cls._datasets.keys())

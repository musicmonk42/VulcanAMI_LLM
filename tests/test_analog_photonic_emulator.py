"""
Comprehensive test suite for analog_photonic_emulator.py
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch

from analog_photonic_emulator import (
    AnalogPhotonicEmulator,
    PhotonicParameters,
    PhotonicNoise,
    WaveguideSimulator,
    MemristorEmulator,
    QuantumPhotonicProcessor,
    HardwareAccelerator,
    PhotonicBackend,
    MultiplexingMode,
    NoiseModel,
    CalibrationData,
    MAX_TENSOR_SIZE,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def photonic_params():
    """Create photonic parameters."""
    return PhotonicParameters()


@pytest.fixture
def emulator():
    """Create emulator."""
    em = AnalogPhotonicEmulator(
        backend=PhotonicBackend.CPU,
        noise_seed=42
    )
    yield em
    em.shutdown()


class TestPhotonicParameters:
    """Test photonic parameters."""
    
    def test_default_parameters(self):
        """Test default parameter creation."""
        params = PhotonicParameters()
        
        assert params.wavelength > 0
        assert params.refractive_index >= 1.0
        assert params.temperature > 0
    
    def test_invalid_wavelength(self):
        """Test invalid wavelength raises error."""
        with pytest.raises(ValueError):
            PhotonicParameters(wavelength=-1)
    
    def test_invalid_refractive_index(self):
        """Test invalid refractive index raises error."""
        with pytest.raises(ValueError):
            PhotonicParameters(refractive_index=0.5)
    
    def test_invalid_temperature(self):
        """Test invalid temperature raises error."""
        with pytest.raises(ValueError):
            PhotonicParameters(temperature=-10)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = PhotonicParameters()
        data = params.to_dict()
        
        assert "wavelength_nm" in data
        assert "temperature_k" in data
        assert "refractive_index" in data


class TestPhotonicNoise:
    """Test photonic noise models."""
    
    def test_shot_noise(self, photonic_params):
        """Test shot noise generation."""
        noise_gen = PhotonicNoise(photonic_params, seed=42)
        signal = np.ones(100, dtype=np.float32)
        
        noisy = noise_gen.shot_noise(signal)
        
        assert noisy.shape == signal.shape
        assert not np.array_equal(noisy, signal)
    
    def test_thermal_noise(self, photonic_params):
        """Test thermal noise generation."""
        noise_gen = PhotonicNoise(photonic_params, seed=42)
        signal = np.ones(100, dtype=np.float32)
        
        noisy = noise_gen.thermal_noise(signal)
        
        assert noisy.shape == signal.shape
    
    def test_phase_noise(self, photonic_params):
        """Test phase noise generation."""
        noise_gen = PhotonicNoise(photonic_params, seed=42)
        signal = np.ones(100, dtype=complex)
        
        noisy = noise_gen.phase_noise(signal)
        
        assert noisy.shape == signal.shape
        assert np.iscomplexobj(noisy)
    
    def test_amplifier_noise(self, photonic_params):
        """Test amplifier noise generation."""
        noise_gen = PhotonicNoise(photonic_params, seed=42)
        signal = np.ones(100, dtype=np.float32)
        
        noisy = noise_gen.amplifier_noise(signal)
        
        assert noisy.shape == signal.shape
    
    def test_realistic_noise(self, photonic_params):
        """Test realistic noise application."""
        noise_gen = PhotonicNoise(photonic_params, seed=42)
        signal = np.ones(100, dtype=np.float32)
        
        noisy = noise_gen.apply_realistic_noise(
            signal,
            [NoiseModel.SHOT, NoiseModel.THERMAL]
        )
        
        assert noisy.shape == signal.shape
        assert not np.array_equal(noisy, signal)


class TestWaveguideSimulator:
    """Test waveguide simulator."""
    
    def test_mode_profile(self, photonic_params):
        """Test mode profile calculation."""
        waveguide = WaveguideSimulator(photonic_params)
        x = np.linspace(-1, 1, 100)
        
        profile = waveguide.mode_profile(x, mode_number=0)
        
        assert profile.shape == x.shape
        assert np.all(profile >= 0)
    
    def test_propagate(self, photonic_params):
        """Test optical propagation."""
        waveguide = WaveguideSimulator(photonic_params)
        field = np.ones(100, dtype=complex)
        
        propagated = waveguide.propagate(field, distance=1e-3)
        
        assert propagated.shape == field.shape
        assert np.iscomplexobj(propagated)
    
    def test_couple(self, photonic_params):
        """Test directional coupler."""
        waveguide = WaveguideSimulator(photonic_params)
        field1 = np.ones(100, dtype=complex)
        field2 = np.ones(100, dtype=complex)
        
        out1, out2 = waveguide.couple(field1, field2, coupling_ratio=0.5)
        
        assert out1.shape == field1.shape
        assert out2.shape == field2.shape
    
    def test_mzi(self, photonic_params):
        """Test Mach-Zehnder interferometer."""
        waveguide = WaveguideSimulator(photonic_params)
        field = np.ones(100, dtype=complex)
        
        output = waveguide.mzi(field, phase_shift=np.pi/2)
        
        assert output.shape == field.shape


class TestMemristorEmulator:
    """Test memristor emulator."""
    
    def test_initialization(self):
        """Test memristor initialization."""
        memristor = MemristorEmulator(rows=10, cols=10)
        
        assert memristor.rows == 10
        assert memristor.cols == 10
        assert memristor.conductance_matrix.shape == (10, 10)
    
    def test_write(self):
        """Test memristor write."""
        memristor = MemristorEmulator(rows=10, cols=10)
        
        memristor.write(5, 5, 0.8)
        
        # Conductance should be affected
        assert memristor.conductance_matrix[5, 5] > 0
    
    def test_write_invalid_indices(self):
        """Test write with invalid indices."""
        memristor = MemristorEmulator(rows=10, cols=10)
        
        with pytest.raises(ValueError):
            memristor.write(15, 5, 0.5)
    
    def test_read(self):
        """Test memristor read."""
        memristor = MemristorEmulator(rows=10, cols=10)
        voltage = np.ones(10)
        
        current = memristor.read(voltage)
        
        assert current.shape == (10,)
    
    def test_read_invalid_size(self):
        """Test read with invalid voltage size."""
        memristor = MemristorEmulator(rows=10, cols=10)
        voltage = np.ones(5)  # Wrong size
        
        with pytest.raises(ValueError):
            memristor.read(voltage)
    
    def test_program_matrix(self):
        """Test programming entire matrix."""
        memristor = MemristorEmulator(rows=10, cols=10)
        matrix = np.random.randn(10, 10)
        
        memristor.program_matrix(matrix)
        
        # All elements should be programmed
        assert np.all(memristor.conductance_matrix > 0)
    
    def test_save_load_state(self, temp_dir):
        """Test saving and loading state."""
        filepath = temp_dir / "memristor.json"
        
        memristor1 = MemristorEmulator(rows=10, cols=10)
        memristor1.write(5, 5, 0.8)
        memristor1.save_state(str(filepath))
        
        assert filepath.exists()
        
        memristor2 = MemristorEmulator(rows=10, cols=10)
        memristor2.load_state(str(filepath))
        
        # State should be restored
        assert memristor2.conductance_matrix[5, 5] == memristor1.conductance_matrix[5, 5]


class TestQuantumPhotonicProcessor:
    """Test quantum photonic processor."""
    
    def test_initialization(self):
        """Test quantum processor initialization."""
        quantum = QuantumPhotonicProcessor(num_modes=4)
        
        assert quantum.num_modes == 4
        assert quantum.state_vector is not None
    
    def test_beam_splitter(self):
        """Test beam splitter gate."""
        quantum = QuantumPhotonicProcessor(num_modes=4)
        
        bs = quantum.beam_splitter(0, 1, reflectivity=0.5)
        
        assert bs.shape == (2, 2)
    
    def test_beam_splitter_invalid_modes(self):
        """Test beam splitter with invalid modes."""
        quantum = QuantumPhotonicProcessor(num_modes=4)
        
        with pytest.raises(ValueError):
            quantum.beam_splitter(0, 10, reflectivity=0.5)
    
    def test_phase_shifter(self):
        """Test phase shifter."""
        quantum = QuantumPhotonicProcessor(num_modes=4)
        
        phase = quantum.phase_shifter(0, np.pi/2)
        
        assert isinstance(phase, complex)
    
    def test_measure(self):
        """Test photon measurement."""
        quantum = QuantumPhotonicProcessor(num_modes=4)
        
        outcome = quantum.measure(0)
        
        assert isinstance(outcome, int)
        assert 0 <= outcome < quantum.fock_cutoff
    
    def test_reset(self):
        """Test state reset."""
        quantum = QuantumPhotonicProcessor(num_modes=4)
        
        # Modify state
        quantum.state_vector[1] = 0.5
        
        quantum.reset()
        
        # Should be back to vacuum
        assert quantum.state_vector[0] == 1.0
        assert quantum.state_vector[1] == 0.0


class TestHardwareAccelerator:
    """Test hardware accelerator."""
    
    def test_initialization(self):
        """Test accelerator initialization."""
        accelerator = HardwareAccelerator(PhotonicBackend.CPU)
        
        assert accelerator.backend == PhotonicBackend.CPU
        accelerator.cleanup()
    
    def test_compile_model(self):
        """Test model compilation."""
        accelerator = HardwareAccelerator(PhotonicBackend.CPU)
        
        model = torch.nn.Linear(10, 5)
        model_hash = accelerator.compile_model(model, (1, 10))
        
        assert model_hash in accelerator.compiled_models
        accelerator.cleanup()
    
    def test_execute_compiled(self):
        """Test executing compiled model."""
        accelerator = HardwareAccelerator(PhotonicBackend.CPU)
        
        model = torch.nn.Linear(10, 5)
        model_hash = accelerator.compile_model(model, (1, 10))
        
        input_data = torch.randn(1, 10)
        output = accelerator.execute(model_hash, input_data)
        
        assert output.shape == (1, 5)
        accelerator.cleanup()


class TestAnalogPhotonicEmulator:
    """Test analog photonic emulator."""
    
    def test_initialization(self):
        """Test emulator initialization."""
        emulator = AnalogPhotonicEmulator(backend=PhotonicBackend.CPU)
        
        assert emulator.backend == PhotonicBackend.CPU
        assert emulator.noise_generator is not None
        emulator.shutdown()
    
    def test_photonic_mvm(self, emulator):
        """Test photonic matrix-vector multiplication."""
        matrix = np.random.randn(10, 20).astype(np.float32)
        vector = np.random.randn(20, 15).astype(np.float32)
        
        result = emulator.emulate_photonic_mvm(matrix, vector)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == matrix.shape[0]
    
    def test_photonic_mvm_oversized(self, emulator):
        """Test rejection of oversized tensors."""
        # Create a 2D tensor that exceeds MAX_TENSOR_SIZE * MAX_TENSOR_SIZE
        # Use a size that's larger than the limit
        huge_size = int(np.sqrt(MAX_TENSOR_SIZE * MAX_TENSOR_SIZE)) + 100
        huge = np.ones((huge_size, huge_size), dtype=np.float32)
        
        with pytest.raises(ValueError):
            emulator._validate_tensor(huge, "huge")
    
    def test_photonic_mvm_with_nan(self, emulator):
        """Test rejection of NaN values."""
        matrix = np.array([[1.0, np.nan], [3.0, 4.0]])
        
        with pytest.raises(ValueError):
            emulator._validate_tensor(matrix, "matrix")
    
    def test_multiplexing_wavelength(self, emulator):
        """Test wavelength multiplexing."""
        tensor = torch.randn(5, 10)
        
        multiplexed = emulator._apply_multiplexing(tensor, MultiplexingMode.WAVELENGTH)
        
        assert multiplexed.shape == tensor.shape
    
    def test_multiplexing_time(self, emulator):
        """Test time multiplexing."""
        tensor = torch.randn(5, 10)
        
        multiplexed = emulator._apply_multiplexing(tensor, MultiplexingMode.TIME)
        
        assert multiplexed.shape == tensor.shape
    
    def test_multiplexing_polarization(self, emulator):
        """Test polarization multiplexing."""
        tensor = torch.randn(5, 10)
        
        multiplexed = emulator._apply_multiplexing(tensor, MultiplexingMode.POLARIZATION)
        
        # Should be complex
        assert torch.is_complex(multiplexed)
    
    def test_calibrate(self, emulator):
        """Test calibration."""
        reference_input = np.random.randn(10, 10).astype(np.float32)
        reference_output = np.random.randn(10, 10).astype(np.float32)
        
        emulator.calibrate(reference_input, reference_output)
        
        assert emulator.calibration.amplitude_corrections is not None
    
    def test_train_in_situ(self, emulator):
        """Test in-situ training."""
        graph = np.random.randn(10, 10).astype(np.float32)
        target = torch.randn(10, device=emulator.device)
        
        loss = emulator.train_in_situ(graph, target, iterations=10)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_get_metrics(self, emulator):
        """Test getting metrics."""
        # Perform operation
        matrix = np.random.randn(5, 10).astype(np.float32)
        vector = np.random.randn(10, 5).astype(np.float32)
        emulator.emulate_photonic_mvm(matrix, vector)
        
        metrics = emulator.get_metrics()
        
        assert "hardware" in metrics
        assert "photonic_mvm" in metrics
    
    def test_reset_metrics(self, emulator):
        """Test resetting metrics."""
        # Perform operation
        matrix = np.random.randn(5, 10).astype(np.float32)
        vector = np.random.randn(10, 5).astype(np.float32)
        emulator.emulate_photonic_mvm(matrix, vector)
        
        emulator.reset_metrics()
        
        metrics = emulator.get_metrics()
        assert "photonic_mvm" not in metrics
    
    def test_save_load_state(self, emulator, temp_dir):
        """Test saving and loading state."""
        # Set up state
        reference = np.random.randn(10, 10).astype(np.float32)
        emulator.calibrate(reference, reference)
        
        # Save
        save_dir = temp_dir / "emulator_state"
        emulator.save_state(str(save_dir))
        
        assert save_dir.exists()
        
        # Create new emulator and load
        emulator2 = AnalogPhotonicEmulator(backend=PhotonicBackend.CPU)
        emulator2.load_state(str(save_dir))
        
        assert emulator2.calibration.amplitude_corrections is not None
        emulator2.shutdown()


class TestCalibrationData:
    """Test calibration data."""
    
    def test_is_valid_fresh(self):
        """Test freshly created calibration is valid."""
        calib = CalibrationData()
        
        assert calib.is_valid(max_age_seconds=3600)
    
    def test_is_valid_expired(self):
        """Test expired calibration is invalid."""
        calib = CalibrationData()
        calib.timestamp = time.time() - 7200  # 2 hours ago
        
        assert not calib.is_valid(max_age_seconds=3600)
    
    def test_validate_corrections(self):
        """Test validation of correction arrays."""
        calib = CalibrationData()
        calib.amplitude_corrections = np.ones((10, 10))
        
        assert calib.validate_corrections((10, 10))
        assert not calib.validate_corrections((5, 5))


class TestBackendSpecific:
    """Test backend-specific functionality."""
    
    def test_memristor_backend(self):
        """Test memristor backend."""
        emulator = AnalogPhotonicEmulator(backend=PhotonicBackend.MEMRISTOR)
        
        assert emulator.memristor is not None
        emulator.shutdown()
    
    def test_quantum_backend(self):
        """Test quantum backend."""
        emulator = AnalogPhotonicEmulator(backend=PhotonicBackend.QUANTUM)
        
        assert emulator.quantum is not None
        emulator.shutdown()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_backend(self):
        """Test CUDA backend."""
        emulator = AnalogPhotonicEmulator(backend=PhotonicBackend.CUDA, device="cuda")
        
        assert emulator.device.type == "cuda"
        emulator.shutdown()


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_no_nan_in_output(self, emulator):
        """Test output doesn't contain NaN."""
        matrix = np.random.randn(10, 10).astype(np.float32)
        vector = np.random.randn(10, 10).astype(np.float32)
        
        result = emulator.emulate_photonic_mvm(matrix, vector)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_noise_doesnt_produce_nan(self, emulator):
        """Test noise application doesn't produce NaN."""
        tensor = torch.randn(10, 10)
        
        noisy = emulator._apply_noise(tensor)
        
        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
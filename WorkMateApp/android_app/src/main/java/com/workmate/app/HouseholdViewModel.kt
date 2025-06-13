/**
 * WorkMate App - Household Vulnerability Prediction
 * 
 * This ViewModel handles the household data input, ML model inference,
 * and data synchronization for the WorkMate mobile application.
 */

package com.workmate.app

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Data class representing household information
 */
data class HouseholdData(
    val householdId: String = "",
    val householdSize: Float = 0f,
    val income: Float = 0f,
    val age: Float = 0f,
    val education: String = "",
    val progressStatus: String = "",
    val region: String = "",
    val programParticipation: String = "",
    val waterAccess: String = "",
    val electricityAccess: String = "",
    val healthcareAccess: String = "",
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Data class for vulnerability prediction results
 */
data class VulnerabilityPrediction(
    val level: VulnerabilityLevel,
    val confidence: Float,
    val probabilities: Map<VulnerabilityLevel, Float>,
    val recommendations: List<String>
)

/**
 * Enum for vulnerability levels
 */
enum class VulnerabilityLevel(val displayName: String, val priority: Int) {
    HIGH("High Risk", 3),
    MODERATE("Moderate Risk", 2),
    LOW("Low Risk", 1);
    
    companion object {
        fun fromIndex(index: Int): VulnerabilityLevel {
            return when(index) {
                0 -> HIGH
                1 -> LOW
                2 -> MODERATE
                else -> LOW
            }
        }
    }
}

/**
 * UI State for the household form
 */
data class HouseholdUiState(
    val householdData: HouseholdData = HouseholdData(),
    val prediction: VulnerabilityPrediction? = null,
    val isLoading: Boolean = false,
    val isOnline: Boolean = false,
    val errorMessage: String? = null,
    val syncStatus: String = "Not synced"
)

/**
 * ViewModel for managing household data and ML predictions
 */
class HouseholdViewModel(
    private val context: Context,
    private val repository: HouseholdRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(HouseholdUiState())
    val uiState: StateFlow<HouseholdUiState> = _uiState

    private var tfliteInterpreter: Interpreter? = null
    private val networkConnectivityHelper = NetworkConnectivityHelper(context)

    init {
        initializeTensorFlowLite()
        observeNetworkConnectivity()
    }

    /**
     * Initialize TensorFlow Lite interpreter with the vulnerability prediction model
     */
    private fun initializeTensorFlowLite() {
        try {
            val modelBuffer = loadModelFile("vulnerability_model.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Optimize for mobile devices
                setUseNNAPI(true) // Use Android Neural Networks API if available
            }
            tfliteInterpreter = Interpreter(modelBuffer, options)
            
            // Verify model input/output shapes
            val inputShape = tfliteInterpreter?.getInputTensor(0)?.shape()
            val outputShape = tfliteInterpreter?.getOutputTensor(0)?.shape()
            
            android.util.Log.d("WorkMate", "Model loaded successfully")
            android.util.Log.d("WorkMate", "Input shape: ${inputShape?.contentToString()}")
            android.util.Log.d("WorkMate", "Output shape: ${outputShape?.contentToString()}")
            
        } catch (e: Exception) {
            android.util.Log.e("WorkMate", "Error loading TensorFlow Lite model", e)
            updateErrorState("Failed to load AI model: ${e.message}")
        }
    }

    /**
     * Load TensorFlow Lite model from assets
     */
    private fun loadModelFile(filename: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Predict vulnerability level for household data
     */
    fun predictVulnerability(householdData: HouseholdData) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)
            
            try {
                val preprocessedInput = preprocessHouseholdData(householdData)
                val prediction = runInference(preprocessedInput)
                
                _uiState.value = _uiState.value.copy(
                    householdData = householdData,
                    prediction = prediction,
                    isLoading = false
                )
                
                // Save household data locally and sync if online
                saveHousehold(householdData, prediction)
                
            } catch (e: Exception) {
                android.util.Log.e("WorkMate", "Error during prediction", e)
                updateErrorState("Prediction failed: ${e.message}")
            }
        }
    }

    /**
     * Preprocess household data for model input
     */
    private fun preprocessHouseholdData(data: HouseholdData): FloatArray {
        // Feature engineering and normalization
        // This should match the preprocessing pipeline used during training
        
        val features = FloatArray(20) // Adjust size based on your model's input requirements
        
        // Numerical features (normalized)
        features[0] = normalizeHouseholdSize(data.householdSize)
        features[1] = normalizeIncome(data.income)
        features[2] = normalizeAge(data.age)
        
        // Categorical features (one-hot encoded)
        val educationEncoded = encodeEducation(data.education)
        val progressStatusEncoded = encodeProgressStatus(data.progressStatus)
        val regionEncoded = encodeRegion(data.region)
        val binaryFeatures = encodeBinaryFeatures(data)
        
        // Combine all features
        System.arraycopy(educationEncoded, 0, features, 3, educationEncoded.size)
        System.arraycopy(progressStatusEncoded, 0, features, 7, progressStatusEncoded.size)
        System.arraycopy(regionEncoded, 0, features, 11, regionEncoded.size)
        System.arraycopy(binaryFeatures, 0, features, 16, binaryFeatures.size)
        
        return features
    }

    /**
     * Normalization functions for numerical features
     */
    private fun normalizeHouseholdSize(size: Float): Float {
        // Normalize household size (assuming mean=4.5, std=2.0)
        return (size - 4.5f) / 2.0f
    }

    private fun normalizeIncome(income: Float): Float {
        // Normalize income (assuming mean=15000, std=8000)
        return (income - 15000f) / 8000f
    }

    private fun normalizeAge(age: Float): Float {
        // Normalize age (assuming mean=45, std=15)
        return (age - 45f) / 15f
    }

    /**
     * One-hot encoding functions for categorical features
     */
    private fun encodeEducation(education: String): FloatArray {
        val encoded = FloatArray(4) // None, Primary, Secondary, Higher
        when (education.lowercase()) {
            "none" -> encoded[0] = 1f
            "primary" -> encoded[1] = 1f
            "secondary" -> encoded[2] = 1f
            "higher" -> encoded[3] = 1f
        }
        return encoded
    }

    private fun encodeProgressStatus(status: String): FloatArray {
        val encoded = FloatArray(4) // Severely Struggling, Struggling, At Risk, On Track
        when (status.lowercase()) {
            "severely struggling" -> encoded[0] = 1f
            "struggling" -> encoded[1] = 1f
            "at risk" -> encoded[2] = 1f
            "on track" -> encoded[3] = 1f
        }
        return encoded
    }

    private fun encodeRegion(region: String): FloatArray {
        val encoded = FloatArray(5) // North, South, East, West, Central
        when (region.lowercase()) {
            "north" -> encoded[0] = 1f
            "south" -> encoded[1] = 1f
            "east" -> encoded[2] = 1f
            "west" -> encoded[3] = 1f
            "central" -> encoded[4] = 1f
        }
        return encoded
    }

    private fun encodeBinaryFeatures(data: HouseholdData): FloatArray {
        val encoded = FloatArray(4)
        encoded[0] = if (data.programParticipation.lowercase() == "yes") 1f else 0f
        encoded[1] = if (data.waterAccess.lowercase() == "yes") 1f else 0f
        encoded[2] = if (data.electricityAccess.lowercase() == "yes") 1f else 0f
        encoded[3] = if (data.healthcareAccess.lowercase() == "yes") 1f else 0f
        return encoded
    }

    /**
     * Run model inference
     */
    private fun runInference(input: FloatArray): VulnerabilityPrediction {
        val interpreter = tfliteInterpreter ?: throw IllegalStateException("Model not loaded")
        
        // Prepare input tensor
        val inputArray = Array(1) { input }
        
        // Prepare output tensor
        val outputArray = Array(1) { FloatArray(3) } // 3 classes: High, Moderate, Low
        
        // Run inference
        interpreter.run(inputArray, outputArray)
        
        val probabilities = outputArray[0]
        val predictedIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[predictedIndex]
        
        // Create probability map
        val probabilityMap = mapOf(
            VulnerabilityLevel.HIGH to probabilities[0],
            VulnerabilityLevel.LOW to probabilities[1],
            VulnerabilityLevel.MODERATE to probabilities[2]
        )
        
        val predictedLevel = VulnerabilityLevel.fromIndex(predictedIndex)
        val recommendations = generateRecommendations(predictedLevel, confidence)
        
        return VulnerabilityPrediction(
            level = predictedLevel,
            confidence = confidence,
            probabilities = probabilityMap,
            recommendations = recommendations
        )
    }

    /**
     * Generate actionable recommendations based on prediction
     */
    private fun generateRecommendations(level: VulnerabilityLevel, confidence: Float): List<String> {
        val recommendations = mutableListOf<String>()
        
        when (level) {
            VulnerabilityLevel.HIGH -> {
                recommendations.addAll(listOf(
                    "Immediate referral to social services required",
                    "Prioritize for emergency assistance programs",
                    "Schedule follow-up visit within 1 week",
                    "Connect with local health services",
                    "Assess immediate food and shelter needs"
                ))
            }
            VulnerabilityLevel.MODERATE -> {
                recommendations.addAll(listOf(
                    "Schedule regular monitoring visits",
                    "Enroll in skill development programs",
                    "Provide information about available support services",
                    "Monitor progress monthly",
                    "Consider preventive interventions"
                ))
            }
            VulnerabilityLevel.LOW -> {
                recommendations.addAll(listOf(
                    "Continue current support level",
                    "Schedule quarterly check-ins",
                    "Provide information about maintaining stability",
                    "Consider graduation from intensive programs"
                ))
            }
        }
        
        if (confidence < 0.7f) {
            recommendations.add("Note: Prediction confidence is low - consider additional assessment")
        }
        
        return recommendations
    }

    /**
     * Save household data and handle online/offline scenarios
     */
    private fun saveHousehold(householdData: HouseholdData, prediction: VulnerabilityPrediction) {
        viewModelScope.launch {
            try {
                // Always save locally first
                repository.saveHouseholdLocally(householdData, prediction)
                
                if (_uiState.value.isOnline) {
                    // Sync to server if online
                    syncToServer(householdData, prediction)
                } else {
                    // Schedule sync for when online
                    scheduleSync()
                    _uiState.value = _uiState.value.copy(syncStatus = "Queued for sync")
                }
                
            } catch (e: Exception) {
                android.util.Log.e("WorkMate", "Error saving household data", e)
                updateErrorState("Failed to save data: ${e.message}")
            }
        }
    }

    /**
     * Sync data to server
     */
    private suspend fun syncToServer(householdData: HouseholdData, prediction: VulnerabilityPrediction) {
        try {
            repository.syncToServer(householdData, prediction)
            _uiState.value = _uiState.value.copy(syncStatus = "Synced")
        } catch (e: Exception) {
            android.util.Log.w("WorkMate", "Sync failed, will retry later", e)
            scheduleSync()
            _uiState.value = _uiState.value.copy(syncStatus = "Sync failed - will retry")
        }
    }

    /**
     * Schedule background sync using WorkManager
     */
    private fun scheduleSync() {
        val syncRequest = OneTimeWorkRequestBuilder<SyncWorker>()
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build()
            )
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                OneTimeWorkRequest.MIN_BACKOFF_MILLIS,
                java.util.concurrent.TimeUnit.MILLISECONDS
            )
            .build()
        
        WorkManager.getInstance(context).enqueue(syncRequest)
    }

    /**
     * Observe network connectivity changes
     */
    private fun observeNetworkConnectivity() {
        viewModelScope.launch {
            networkConnectivityHelper.isConnected.collect { isConnected ->
                _uiState.value = _uiState.value.copy(isOnline = isConnected)
                
                if (isConnected) {
                    // Trigger sync of pending data
                    triggerPendingSync()
                }
            }
        }
    }

    /**
     * Trigger sync of all pending offline data
     */
    private fun triggerPendingSync() {
        viewModelScope.launch {
            try {
                val pendingCount = repository.syncPendingData()
                if (pendingCount > 0) {
                    _uiState.value = _uiState.value.copy(
                        syncStatus = "Synced $pendingCount records"
                    )
                }
            } catch (e: Exception) {
                android.util.Log.e("WorkMate", "Error syncing pending data", e)
            }
        }
    }

    /**
     * Update UI state with error message
     */
    private fun updateErrorState(message: String) {
        _uiState.value = _uiState.value.copy(
            isLoading = false,
            errorMessage = message
        )
    }

    /**
     * Clear error message
     */
    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }

    /**
     * Reset form data
     */
    fun resetForm() {
        _uiState.value = _uiState.value.copy(
            householdData = HouseholdData(),
            prediction = null,
            errorMessage = null
        )
    }

    override fun onCleared() {
        super.onCleared()
        tfliteInterpreter?.close()
    }
}

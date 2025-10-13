"""Comment data models"""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Sentiment(str, Enum):
    """Comment sentiment categories."""
    FOR = "for"
    AGAINST = "against"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class Topic(str, Enum):
    """Comment topic categories."""
    REIMBURSEMENT_PAYMENT = "reimbursement_payment"
    COST_FINANCIAL = "cost_financial"
    SERVICE_COVERAGE = "service_coverage"
    ACCESS_TO_CARE = "access_to_care"
    WORKFORCE_STAFFING = "workforce_staffing"
    METHODOLOGY_MEASUREMENT = "methodology_measurement"
    IMPLEMENTATION_FEASIBILITY = "implementation_feasibility"
    ADMINISTRATIVE_BURDEN = "administrative_burden"
    TELEHEALTH_DIGITAL = "telehealth_digital"
    HEALTH_EQUITY = "health_equity"
    QUALITY_PROGRAMS = "quality_programs"
    LEGAL_CLARITY = "legal_clarity"
    UNCLEAR = "unclear"


class Category(str, Enum):
    """Commenter categories."""
    PHYSICIANS_SURGEONS = "Physicians & Surgeons"
    OTHER_LICENSED_CLINICIANS = "Other Licensed Clinicians"
    HEALTHCARE_PRACTICE_STAFF = "Healthcare Practice Staff"
    PATIENTS_CAREGIVERS = "Patients & Caregivers"
    PATIENT_ADVOCATES = "Patient/Disability Advocates & Advocacy Organizations"
    PROFESSIONAL_ASSOCIATIONS = "Professional Associations"
    HOSPITALS_HEALTH_SYSTEMS = "Hospitals Health Systems & Networks"
    HEALTHCARE_COMPANIES = "Healthcare Companies & Corporations"
    PHARMA_BIOTECH = "Pharmaceutical & Biotech Companies"
    MEDICAL_DEVICE_DIGITAL_HEALTH = "Medical Device & Digital Health Companies"
    GOVERNMENT_PUBLIC_PROGRAMS = "Government & Public Programs"
    ACADEMIC_RESEARCH = "Academic & Research Institutions"
    NONPROFITS_FOUNDATIONS = "Nonprofits & Foundations"
    INDIVIDUALS_PRIVATE_CITIZENS = "Individuals / Private Citizens"
    ANONYMOUS_NOT_SPECIFIED = "Anonymous / Not Specified"


class DoctorSpecialization(str, Enum):
    """Medical doctor (MD/DO) specializations based on ABMS board certifications."""
    # Primary Care & General Specialties
    FAMILY_MEDICINE = "family_medicine"
    INTERNAL_MEDICINE = "internal_medicine"
    PEDIATRICS = "pediatrics"
    GENERAL_PRACTICE = "general_practice"

    # Surgical Specialties
    GENERAL_SURGERY = "general_surgery"
    CARDIOTHORACIC_SURGERY = "cardiothoracic_surgery"
    COLON_RECTAL_SURGERY = "colon_rectal_surgery"
    NEUROLOGICAL_SURGERY = "neurological_surgery"
    ORTHOPEDIC_SURGERY = "orthopedic_surgery"
    OTOLARYNGOLOGY = "otolaryngology"  # ENT
    PLASTIC_SURGERY = "plastic_surgery"
    UROLOGY = "urology"
    VASCULAR_SURGERY = "vascular_surgery"

    # Medical Specialties
    ALLERGY_IMMUNOLOGY = "allergy_immunology"
    ANESTHESIOLOGY = "anesthesiology"
    CARDIOLOGY = "cardiology"
    CRITICAL_CARE_MEDICINE = "critical_care_medicine"
    DERMATOLOGY = "dermatology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    ENDOCRINOLOGY = "endocrinology"
    GASTROENTEROLOGY = "gastroenterology"
    GERIATRICS = "geriatrics"
    HEMATOLOGY = "hematology"
    INFECTIOUS_DISEASE = "infectious_disease"
    NEPHROLOGY = "nephrology"
    ONCOLOGY = "oncology"
    PULMONOLOGY = "pulmonology"
    RHEUMATOLOGY = "rheumatology"
    SLEEP_MEDICINE = "sleep_medicine"
    SPORTS_MEDICINE = "sports_medicine"

    # Hospital-Based Specialties
    HOSPITALIST = "hospitalist"
    INTENSIVIST = "intensivist"
    PATHOLOGY = "pathology"
    RADIOLOGY = "radiology"
    NUCLEAR_MEDICINE = "nuclear_medicine"

    # Specialized Medical Fields
    OBSTETRICS_GYNECOLOGY = "obstetrics_gynecology"
    OPHTHALMOLOGY = "ophthalmology"
    PAIN_MEDICINE = "pain_medicine"
    PHYSICAL_MEDICINE_REHABILITATION = "physical_medicine_rehabilitation"
    PREVENTIVE_MEDICINE = "preventive_medicine"
    PSYCHIATRY = "psychiatry"

    # Radiation & Imaging
    RADIATION_ONCOLOGY = "radiation_oncology"
    INTERVENTIONAL_RADIOLOGY = "interventional_radiology"

    # Subspecialties Commonly Seen
    RETINA_SPECIALIST = "retina_specialist"  # Ophthalmology subspecialty
    INTERVENTIONAL_CARDIOLOGY = "interventional_cardiology"
    ELECTROPHYSIOLOGY = "electrophysiology"
    MATERNAL_FETAL_MEDICINE = "maternal_fetal_medicine"
    NEONATOLOGY = "neonatology"
    PEDIATRIC_SUBSPECIALTY = "pediatric_subspecialty"  # Various pediatric subspecialties

    # Other
    OSTEOPATHIC_MEDICINE = "osteopathic_medicine"  # DO with no specific specialty
    UNSPECIFIED = "unspecified"
    OTHER = "other"


class LicensedProfessionalType(str, Enum):
    """Non-physician licensed healthcare professional types."""
    # Advanced Practice Nursing
    NURSE_PRACTITIONER = "nurse_practitioner"  # NP, FNP, etc.
    CERTIFIED_NURSE_ANESTHETIST = "certified_nurse_anesthetist"  # CRNA
    CERTIFIED_NURSE_MIDWIFE = "certified_nurse_midwife"  # CNM
    CLINICAL_NURSE_SPECIALIST = "clinical_nurse_specialist"  # CNS

    # Nursing
    REGISTERED_NURSE = "registered_nurse"  # RN
    LICENSED_PRACTICAL_NURSE = "licensed_practical_nurse"  # LPN/LVN
    CERTIFIED_NURSING_ASSISTANT = "certified_nursing_assistant"  # CNA

    # Physician Assistant
    PHYSICIAN_ASSISTANT = "physician_assistant"  # PA

    # Therapy Professions
    PHYSICAL_THERAPIST = "physical_therapist"  # PT
    PHYSICAL_THERAPIST_ASSISTANT = "physical_therapist_assistant"  # PTA
    OCCUPATIONAL_THERAPIST = "occupational_therapist"  # OT
    OCCUPATIONAL_THERAPY_ASSISTANT = "occupational_therapy_assistant"  # OTA
    SPEECH_LANGUAGE_PATHOLOGIST = "speech_language_pathologist"  # SLP
    RESPIRATORY_THERAPIST = "respiratory_therapist"  # RT
    RECREATIONAL_THERAPIST = "recreational_therapist"
    MASSAGE_THERAPIST = "massage_therapist"  # LMT

    # Optometry & Vision Care
    OPTOMETRIST = "optometrist"  # OD
    OPHTHALMIC_TECHNICIAN = "ophthalmic_technician"
    OPHTHALMIC_ASSISTANT = "ophthalmic_assistant"
    CERTIFIED_OPHTHALMIC_ASSISTANT = "certified_ophthalmic_assistant"  # COA
    OPTOMETRIC_TECHNICIAN = "optometric_technician"

    # Pharmacy
    PHARMACIST = "pharmacist"  # PharmD, RPh
    PHARMACY_TECHNICIAN = "pharmacy_technician"

    # Dental Professions
    DENTIST = "dentist"  # DDS/DMD
    DENTAL_HYGIENIST = "dental_hygienist"
    DENTAL_ASSISTANT = "dental_assistant"

    # Podiatry
    PODIATRIST = "podiatrist"  # DPM

    # Chiropractic
    CHIROPRACTOR = "chiropractor"  # DC

    # Medical Laboratory & Diagnostic
    MEDICAL_LABORATORY_SCIENTIST = "medical_laboratory_scientist"  # MLS
    MEDICAL_LABORATORY_TECHNICIAN = "medical_laboratory_technician"  # MLT
    PHLEBOTOMIST = "phlebotomist"
    CYTOTECHNOLOGIST = "cytotechnologist"
    HISTOTECHNOLOGIST = "histotechnologist"

    # Radiology & Imaging Technologists
    RADIOLOGIC_TECHNOLOGIST = "radiologic_technologist"
    MRI_TECHNOLOGIST = "mri_technologist"
    CT_TECHNOLOGIST = "ct_technologist"
    ULTRASOUND_TECHNOLOGIST = "ultrasound_technologist"  # Sonographer
    NUCLEAR_MEDICINE_TECHNOLOGIST = "nuclear_medicine_technologist"
    RADIATION_THERAPIST = "radiation_therapist"
    CARDIOVASCULAR_TECHNOLOGIST = "cardiovascular_technologist"

    # Allied Health Professionals
    DIETITIAN = "dietitian"  # RD, RDN
    NUTRITIONIST = "nutritionist"
    SOCIAL_WORKER = "social_worker"  # LCSW, LSW
    CASE_MANAGER = "case_manager"
    GENETIC_COUNSELOR = "genetic_counselor"
    AUDIOLOGIST = "audiologist"

    # Emergency & Surgical Support
    PARAMEDIC = "paramedic"  # EMT-P
    EMERGENCY_MEDICAL_TECHNICIAN = "emergency_medical_technician"  # EMT
    SURGICAL_TECHNOLOGIST = "surgical_technologist"  # CST
    SURGICAL_ASSISTANT = "surgical_assistant"

    # Medical Assistants & Support Staff
    MEDICAL_ASSISTANT = "medical_assistant"  # CMA, RMA
    MEDICAL_SCRIBE = "medical_scribe"
    PATIENT_CARE_TECHNICIAN = "patient_care_technician"

    # Mental Health Professionals
    LICENSED_PROFESSIONAL_COUNSELOR = "licensed_professional_counselor"  # LPC
    LICENSED_CLINICAL_SOCIAL_WORKER = "licensed_clinical_social_worker"  # LCSW
    MARRIAGE_FAMILY_THERAPIST = "marriage_family_therapist"  # LMFT
    LICENSED_MENTAL_HEALTH_COUNSELOR = "licensed_mental_health_counselor"  # LMHC
    PSYCHOLOGIST = "psychologist"  # PhD, PsyD
    PSYCHIATRIC_TECHNICIAN = "psychiatric_technician"
    BEHAVIOR_ANALYST = "behavior_analyst"  # BCBA

    # Specialized Technicians & Coordinators
    EKG_TECHNICIAN = "ekg_technician"
    SLEEP_TECHNOLOGIST = "sleep_technologist"
    PERFUSIONIST = "perfusionist"
    ORTHOTIST_PROSTHETIST = "orthotist_prosthetist"
    CLINICAL_RESEARCH_COORDINATOR = "clinical_research_coordinator"

    # Other
    UNSPECIFIED = "unspecified"
    OTHER = "other"


class Comment(BaseModel):
    """Comment model."""

    id: str
    document_id: str
    comment_text: Optional[str] = None
    category: Optional[str] = None
    sentiment: Optional[str] = None
    topics: Optional[List[str]] = None
    doctor_specialization: Optional[str] = None
    licensed_professional_type: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None
    posted_date: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class CommentData(BaseModel):
    """Simple comment data for processing."""

    id: str
    comment_text: str = ""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: tuple) -> "CommentData":
        """Create CommentData from database row.

        Args:
            row: Database row (id, comment_text, first_name, last_name, organization)

        Returns:
            CommentData instance
        """
        return cls(
            id=row[0],
            comment_text=row[1] or "",
            first_name=row[2],
            last_name=row[3],
            organization=row[4],
        )


class CommentClassification(BaseModel):
    """Structured output for comment classification."""

    category: Category = Field(
        description="The category of the commenter based on their role and affiliation"
    )
    sentiment: Sentiment = Field(
        description="The overall sentiment of the comment toward the regulation"
    )
    topics: List[Topic] = Field(
        description="The topics discussed in the comment (can be multiple topics)"
    )
    doctor_specialization: Optional[DoctorSpecialization] = Field(
        default=None,
        description="The medical specialization of the doctor (ONLY if category is 'Physicians & Surgeons')"
    )
    licensed_professional_type: Optional[LicensedProfessionalType] = Field(
        default=None,
        description="The type of licensed professional (ONLY if category is 'Other Licensed Clinicians')"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decisions"
    )

    @field_validator('doctor_specialization')
    @classmethod
    def validate_doctor_specialization(cls, v, info):
        """Ensure doctor_specialization is only set when category is Physicians & Surgeons."""
        if v is not None:
            category = info.data.get('category')
            if category != Category.PHYSICIANS_SURGEONS:
                raise ValueError(
                    f"doctor_specialization can only be set when category is 'Physicians & Surgeons', "
                    f"but category is '{category.value if category else None}'"
                )
        return v

    @field_validator('licensed_professional_type')
    @classmethod
    def validate_licensed_professional_type(cls, v, info):
        """Ensure licensed_professional_type is only set when category is Other Licensed Clinicians."""
        if v is not None:
            category = info.data.get('category')
            if category != Category.OTHER_LICENSED_CLINICIANS:
                raise ValueError(
                    f"licensed_professional_type can only be set when category is 'Other Licensed Clinicians', "
                    f"but category is '{category.value if category else None}'"
                )
        return v

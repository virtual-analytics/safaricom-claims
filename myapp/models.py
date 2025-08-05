from django.contrib.auth.models import User
from django.db import models

# =======================
# Claim Record Model
# =======================

class ClaimRecord(models.Model):
    """
    Stores a single claim entry with detailed financial and medical claim information.
    """
    admit_id = models.CharField(max_length=32)
    icd10_code = models.CharField(max_length=32)
    encounter = models.CharField(max_length=64)
    service_code = models.CharField(max_length=64)
    quantity = models.IntegerField()
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    service_id = models.CharField(max_length=32)
    benefit = models.CharField(max_length=128)
    benefit_desc = models.CharField(max_length=256)
    claim_prov_date = models.DateField()
    claim_pod = models.CharField(max_length=16)
    dob = models.DateField()
    gender = models.CharField(max_length=8)
    dependent_type = models.CharField(max_length=32)
    ailment = models.CharField(max_length=128)
    claim_me = models.CharField(max_length=64)
    claim_ce = models.CharField(max_length=64)
    claim_in_global = models.CharField(max_length=64)
    claim_pa = models.CharField(max_length=64)
    claim_pr = models.CharField(max_length=64)
    prov_name = models.CharField(max_length=128)
    pol_id = models.CharField(max_length=32)
    pol_name = models.CharField(max_length=128)
    cost_center = models.CharField(max_length=128)

    def __str__(self):
        return f"Claim {self.admit_id} - {self.service_code} - {self.amount}"

    class Meta:
        verbose_name = "Claim Record"
        verbose_name_plural = "Claim Records"
        ordering = ['-claim_prov_date']


# =======================
# User Role Management
# =======================

class UserProfile(models.Model):
    """
    Extends the built-in User model with roles for role-based access control.
    """
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('safaricom', 'Safaricom'),
        ('manager', 'Manager'),
        ('viewer', 'Viewer'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='viewer')

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"
